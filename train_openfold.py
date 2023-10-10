#! /usr/bin/env python
import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch

from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    DummyDataLoader,
)
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants
from openfold.utils.argparse import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_global_step_from_zero_checkpoint
)

from openfold.utils.logger import PerformanceLoggingCallback

import sidechainnet.examples.losses as scn_losses
from sidechainnet.research.openfold.openfold_loss import OpenMMLR


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, use_wandb=False, log_other_metrics_on_step=False):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        if config.loss.openmm.use_openmm_warmup:
            self.openmm_scheduler = OpenMMLR(steps=config.loss.openmm.openmm_warmup_steps)
        else:
            self.openmm_scheduler = None
        self.model = AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss, self.openmm_scheduler, use_wandb=use_wandb)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1

        self.phase = None
        self.log_other_metrics_on_step = log_other_metrics_on_step

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = self.phase
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(
                    not train or self.config.loss.openmm.add_struct_metrics)
            )

        for k,v in other_metrics.items():
            try:
                mean = torch.mean(v)
            except TypeError:
                # Structure metrics can be numpy arrays
                mean = np.mean(v)
            self.log(
                f"{phase}/{k}",
                mean,
                on_step=self.log_other_metrics_on_step, on_epoch=True, logger=True
            )

    def training_step(self, batch, batch_idx):
        self.phase = self.loss.mode = "train"
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs)

        # Step OpenMM-Loss scheduler if in use
        if self.openmm_scheduler is not None:
            self.openmm_scheduler.step(self.global_step)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        self.loss.mode = 'val' if self.phase != 'test' else 'test'
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
       
        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss and other metrics
        batch["use_clamped_fape"] = 0.
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
        return loss_breakdown

    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            try:
                superimposed_pred, alignment_rmsd = superimpose(
                    gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
                )
                gdt_ts_score = gdt_ts(
                    superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                )
                gdt_ha_score = gdt_ha(
                    superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                )

                metrics["rmsd_ca"] = alignment_rmsd
                metrics["gdtts_ca"] = gdt_ts_score
                metrics["gdtha_ca"] = gdt_ha_score

                if gt_coords.shape[0] != 1:
                    metrics["rmsd_ca"] = torch.mean(alignment_rmsd)

                # Record various structure metrics, only supports batch size = 1
                if gt_coords.shape[0] == 1:
                    # Create our superimposed and de-padded all-atom variables for analysis
                    flat_gt = gt_coords_masked.reshape(gt_coords.shape[0], -1, 3)
                    flat_pred = pred_coords_masked.reshape(pred_coords.shape[0], -1, 3)
                    flat_all_atom_mask = all_atom_mask.reshape(all_atom_mask.shape[0], -1)

                    flat_gt_unpadded = flat_gt[flat_all_atom_mask.bool()]
                    flat_pred_unpadded = flat_pred[flat_all_atom_mask.bool()]
                    flat_gt_unpadded_np = flat_gt_unpadded.cpu().numpy()

                    # >>> All-atom RMSD
                    flat_superimposed_pred_aa, rmsd_all = superimpose(
                        flat_gt, flat_pred, flat_all_atom_mask)
                    metrics["rmsd_aa"] = rmsd_all
                    flat_superimposed_pred_aa_unpadded_np = flat_superimposed_pred_aa[
                        flat_all_atom_mask.bool()].cpu().numpy()

                    # >>> Global Metrics (GDC_all, TM score)
                    gdcall_aa = scn_losses.gdc_all(flat_gt_unpadded_np,
                                                flat_superimposed_pred_aa_unpadded_np,
                                                skip_alignment=True,
                                                as_percent=False)
                    tmscore_aa = scn_losses.tm_score(
                        flat_gt_unpadded_np,
                        flat_superimposed_pred_aa_unpadded_np,
                        skip_alignment=True)
                    tmscore_ca = scn_losses.tm_score(
                        gt_coords_masked_ca[all_atom_mask_ca.bool()].cpu().numpy(),
                        superimposed_pred[all_atom_mask_ca.bool()].cpu().numpy(),
                        skip_alignment=True)

                    # >>> Local Metrics (DRMSD, LDDT, no alignment required)
                    # Note: Current implementation of lddt all-atom in OpenFold does not
                    # exclude atoms within the same residue. Use SidechainNet impl. instead.
                    drmsd_aa = scn_losses.drmsd(flat_gt_unpadded, flat_pred_unpadded)
                    lddt_aa = scn_losses.lddt_all(
                        flat_gt.reshape(-1, 3) * all_atom_mask.reshape(-1, 1),
                        flat_pred.reshape(-1, 3) * all_atom_mask.reshape(-1, 1),
                        atom_mask=all_atom_mask.reshape(-1).bool(),
                        residue_shape=gt_coords.shape[-1],
                        cutoff=15)

                    metrics["gdcall_aa"] = gdcall_aa
                    metrics["tmscore_aa"] = tmscore_aa
                    metrics["tmscore_ca"] = tmscore_ca
                    metrics["drmsd_aa"] = drmsd_aa
                    metrics["lddt_aa"] = lddt_aa
            except ZeroDivisionError:
                # An alignment error occurred, so we skip superimposition metrics
                pass
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
#        return torch.optim.Adam(
#            self.model.parameters(),
#            lr=learning_rate,
#            eps=eps
#        )
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        if(not self.model.template_config.enabled):
            ema["params"] = {k:v for k,v in ema["params"].items() if not "template" in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step
        # Load the openmm scheduler step, if used
        if self.openmm_scheduler is None:
            return
        self.openmm_scheduler.cur_step = lr_step
        cur_openmm = self.openmm_scheduler.get_lr()
        logging.warning(f"Resuming from lr step: {lr_step} with current openmm weight"
                        f" = {cur_openmm}".format(lr_step))

    def load_from_jax(self, jax_path):
        model_basename = os.path.splitext(
                os.path.basename(
                    os.path.normpath(jax_path)
                )
        )[0]
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(
                self.model, jax_path, version=model_version
        )


def update_openmm_config(config, args):
    """Update training config in-place with OpenMM-Loss training arguments."""
    config.loss.openmm.use_openmm = args.use_openmm
    config.loss.openmm.add_struct_metrics = args.add_struct_metrics
    config.loss.openmm.weight = args.openmm_weight
    config.loss.openmm.write_pdbs_every_n_steps = args.write_pdbs_every_n_steps
    config.loss.openmm.pdb_dir = os.path.join(args.output_dir, "pdbs")
    if args.use_openmm_warmup:
        config.loss.openmm.use_openmm_warmup = True
    if args.openmm_warmup_steps is not None:
        config.loss.openmm.openmm_warmup_steps = args.openmm_warmup_steps


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=(str(args.precision) == "16")
    ) 

    # Update config with user-specified arguments for OpenMM-Loss
    update_openmm_config(config, args)

    model_module = OpenFoldWrapper(config, args.wandb, args.log_other_metrics_on_step)
    if(args.resume_from_ckpt and "finetuning_" in args.resume_from_ckpt):
        logging.info("Loading weights from OpenFold checkpoint...")
        sd = torch.load(args.resume_from_ckpt)
        model_module.resume_last_lr_step(0)
        logging.info("Successfully loaded last lr step...")
        logging.info("Adding model. prefix to OpenFold state dict...")
        sd = {"model." + k: v for k, v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")
    elif(args.resume_from_ckpt and not args.resume_model_weights_only):
        if(os.path.isdir(args.resume_from_ckpt)):  
            last_global_step = get_global_step_from_zero_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt)
            try:
                last_global_step = int(sd['global_step'])
            except KeyError:
                last_global_step = 0
        model_module.resume_last_lr_step(last_global_step)
        logging.info("Successfully loaded last lr step...")
    elif(args.resume_from_ckpt and args.resume_model_weights_only):
        if(os.path.isdir(args.resume_from_ckpt)):
            sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt)
        sd = {k[len("module."):]:v for k,v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")
    if(args.resume_from_jax_params):
        model_module.load_from_jax(args.resume_from_jax_params)
        logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")
 
    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    #data_module = DummyDataLoader("new_batch.pickle")
    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    if(args.checkpoint_every_epoch):
        mc = ModelCheckpoint(
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_top_k=-1,
        )
        callbacks.append(mc)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val/lddt_ca",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    if(args.wandb):
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity}
        )
        # Save arguments to wandb config
        wdb_logger.experiment.config.update(vars(args), allow_val_change=True)
        loggers.append(wdb_logger)
        logging.info("WandB experiment dir:", wdb_logger.experiment.dir)

    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
        )
        if(args.wandb):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = None
 
    if(args.wandb):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.output_dir,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="""Whether to checkpoint at the end of every training epoch"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_from_jax_params", type=str, default=None,
        help="""Path to an .npz JAX parameter file with which to initialize the model"""
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--log_other_metrics_on_step", type=bool_type, default=False,
        help="Whether to log auxilliary metrics on every step."
    )

    # Arguments for OpenMM-Loss Training
    omm_loss = parser.add_argument_group("OpenMM-Loss Training")
    omm_loss.add_argument(
        "--use_openmm", type=bool_type, default=False, help="Whether to use OpenMM loss."
    )
    omm_loss.add_argument(
        "--openmm_weight", type=float, default=0.01, help="Weight applied to OpenMM loss."
    )
    omm_loss.add_argument(
        "--add_struct_metrics", type=bool_type, default=True, help="Whether to add "
        "additional structure metrics to wandb including RMSD, GDC, DRMSD, LDDT, etc."
    )
    omm_loss.add_argument(
        "--write_pdbs_every_n_steps", type=int, default=-1, help="Frequency with which to"
        " write pdbs of the predicted structures.")
    omm_loss.add_argument(
        "--use_openmm_warmup", type=bool_type, default=False,
        help="Whether to use OpenMM LR warmup. Scales the value of OpenMM"
        " linearly from 1e-4 to 1 over 1000 steps..")
    omm_loss.add_argument(
        "--openmm_warmup_steps", type=int, default=1000,
        help="Number of steps to warmup OpenMM."
    )
    

    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser, 
        [
            "--accelerator", 
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_epoch",
            "--reload_dataloaders_every_n_epochs",
        ]
    ) 

    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if(str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    if(args.resume_from_jax_params is not None and args.resume_from_ckpt is not None):
        raise ValueError("Choose between loading pretrained Jax-weights and a checkpoint-path")

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)
