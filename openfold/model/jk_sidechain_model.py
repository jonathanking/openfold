"""This file contains a model meant to replace the sidechain component of AlphaFold2.

To match the input/output of AlphaFold2's sidechain ResNet, the following are required:
    - module.forward input
        s: [*, C_hidden], single embedding
        s_initial: [*, C_hidden], single embedding at start of structure module
    - module.forward output
        unnormalized_s: [*, no_angles, 2], unnormalized torsion angles
        s: [*, no_angles, 2] torsion angles

When initializing the model, the following are utilized by the ResNet and thus may be
useful to have as arguments:
    - module.init (has access to these, and more)
        c_s: input channel size
        c_hidden: hidden channel size, for resnet interior
        no_blocks: number of resnet blocks
        no_angles: number of torsion angles to predict
        epsilon: small value to add to denominator of normalization

To match some of the design decisions of AlphaFold2, the following are suggested:
    - Use Openfold's Linear layers that specify DeepMind's initialization strategies
    - Perform relu to s and s_initial inputs

The model on this page will be based on the transformer encoder module, not a ResNet.
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from openfold.model.primitives import Linear

from sidechainnet.examples.transformer import PositionalEncoding


class AngleTransformer(nn.Module):
    """A shallow Transformer Encoder to predict torsion angles from AF2 seq embeding."""
    def __init__(self,
                 c_s,
                 c_hidden,
                 no_blocks,
                 no_angles,
                 epsilon,
                 dropout=0.1,
                 d_ff=2048,
                 no_heads=4,
                 activation='relu'):
        super().__init__()
        self.eps = epsilon

        self.linear_initial = Linear(c_s, c_hidden)
        self.linear_in = Linear(c_s, c_hidden)
        self.relu = nn.ReLU()

        self.pos_encoder = PositionalEncoding(c_hidden)
        encoder_layers = TransformerEncoderLayer(c_hidden,
                                                 nhead=no_heads,
                                                 dim_feedforward=d_ff,
                                                 dropout=dropout,
                                                 activation=activation,
                                                 batch_first=True,
                                                 norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=no_blocks)

        self.linear_out = nn.Linear(c_hidden, no_angles * 2)

    def forward(self, s, s_initial):
        # [*, C_hidden], eg [1, 256 (L), 384  (c_s)]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        # Transformer in lieu of ResNet
        s = self.pos_encoder(s)
        s = self.transformer_encoder(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            ))
        s = s / norm_denom

        return unnormalized_s, s  # Batch x Length x 7 x 2
