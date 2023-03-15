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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from deepspeed.ops.adam import DeepSpeedCPUAdam

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
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.script_utils import get_model_basename
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
import wandb

from tqdm import tqdm
from tqdm.utils import _unicode, disp_len

_TQDM_STATUS_EVERY_N = 2

if "SLURM_JOB_ID" in os.environ:
    def status_printer(self, file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        self._status_printer_counter = 0
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda: None)()
            getattr(sys.stdout, 'flush', lambda: None)()

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            self._status_printer_counter += 1
            if self._status_printer_counter % _TQDM_STATUS_EVERY_N == 0:
                len_s = disp_len(s)
                fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n')
                last_len[0] = len_s

        return print_status
    tqdm.status_printer = status_printer
    logging.info("Overwriting tqdm status printer for SLURM output.")


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1

        self.automatic_optimization = not self.config.model.disable_backwards

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
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
                superimposition_metrics=True  # MOD-JK: Changed to compute gdtts on train
            )

        for k, v in other_metrics.items():
            if not train and k == "lddt_ca":
                print(f"lddt_ca: {v.item()}")
            self.log(f"{phase}/{k}", v, on_step=train, on_epoch=True, logger=True)
        
        if self.config.logging.log_to_csv:
            self._log_to_csv(batch, loss_breakdown, other_metrics, train=train)
    
    def _log_to_csv(self, batch, loss_breakdown, other_metrics, train=True):
        """Log to a CSV file for later analysis.
        Each row should reflect metric for a single protein (there may be multiple
        proteins in a batch). Each row should have the protein name followed by all of the
        loss and metric values found in the loss_breakdown and other_metric dictionaries.
        """
        phase = "train" if train else "val"
        csv_path = os.path.join(self.logger.experiment[0].dir, f"{phase}.csv")
        if not os.path.exists(csv_path):
            logging.info(f"Creating CSV file at {csv_path}")
            with open(csv_path, "w") as f:
                f.write("protein_name,")
                f.write(",".join(loss_breakdown.keys()))
                f.write(",")
                f.write(",".join(other_metrics.keys()))
                f.write("\n")

        protein_name = batch["name"][0] if len(batch["name"]) == 1 else "batch"

        fmt = lambda tensor_to_str: str(tensor_to_str.item()) if isinstance(tensor_to_str, torch.Tensor) else str(tensor_to_str)

        with open(csv_path, "a") as f:
            f.write(f"{protein_name},")
            f.write(",".join([fmt(v) for v in loss_breakdown.values()]))
            f.write(",")
            f.write(",".join([fmt(v) for v in other_metrics.values()]))
            f.write("\n")

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()

        if batch['all_atom_positions'].sum() == 0:
            # TODO-JK: What happened here?
            print(f"Protein {batch['name']} has all 0 gt-atoms. Skipping.")
            return torch.tensor(0., requires_grad=True, device=self.device)

        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        if not self.config.model.disable_backwards:
            outputs = self(batch)
        else:
            with torch.no_grad():
                outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        if outputs['final_atom_positions'].sum() == 0:
            # TODO-JK: What happened here?
            print(f"Protein {batch['name']} has all 0 pred atoms. Skipping.")
            return torch.tensor(0., requires_grad=True, device=self.device)

        # Compute loss
        if not self.config.model.disable_backwards:
            loss, loss_breakdown = self.loss(
                outputs, batch, _return_breakdown=True
            )
        else:
            with torch.no_grad():
                loss, loss_breakdown = self.loss(
                    outputs, batch, _return_breakdown=True
                )

        # Log it
        self._log(loss_breakdown, batch, outputs)

        # Backprop
        if not self.config.model.disable_backwards and not self.automatic_optimization:
            self.manual_backward(loss)
            if self.config.model.grad_clip_val != 0:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.config.model.grad_clip_val
                )
            opt.step()
            self.lr_schedulers().step()

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
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
        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None
    
    def on_fit_start(self):
        self._fit_start_time = time.time()

    def on_fit_end(self):
        self._fit_end_time = time.time()
        wandb.run.summary["total_fit_time"] = self._fit_end_time - self._fit_start_time
    
    def on_train_batch_start(self, batch, batch_idx):
        # Log the training_batch_time
        self._batch_start_time = time.time()
    
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # Compute and log the training_batch_time
        batch_end_time = time.time()
        batch_duration = batch_end_time - self._batch_start_time
        self.log("train/batch_time", batch_duration, on_step=True, on_epoch=False, logger=True)

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

        if batch["all_atom_positions"].shape[0] != 1:
            metrics["lddt_ca"] = torch.mean(lddt_ca_score)

        if batch["all_atom_positions"].shape[0] == 1:
            # Doesn't work for batch size > 1
            drmsd_ca_score = drmsd(
                pred_coords_masked_ca,
                gt_coords_masked_ca,
                mask=all_atom_mask_ca, # still required here to compute n
            )
            metrics["drmsd_ca"] = drmsd_ca_score

        if (superimposition_metrics and self.config.loss.openmm.add_struct_metrics):
            # Original code (alpha-code analysis)
            try:
                superimposed_pred, alignment_rmsd = superimpose(
                    gt_coords_masked_ca,
                    pred_coords_masked_ca,
                    all_atom_mask_ca,
                )
                gdt_ts_score = gdt_ts(superimposed_pred, gt_coords_masked_ca,
                                    all_atom_mask_ca)
                gdt_ha_score = gdt_ha(superimposed_pred, gt_coords_masked_ca,
                                    all_atom_mask_ca)
                metrics["rmsd_ca"] = alignment_rmsd
                metrics["gdtts_ca"] = gdt_ts_score
                metrics["gdtha_ca"] = gdt_ha_score

                if gt_coords.shape[0] != 1:
                    metrics["rmsd_ca"] = torch.mean(alignment_rmsd)

                # MOD-JK heavily added code, only supports batch size of one at the moment
                if gt_coords.shape[0] == 1:
                    # Create our superimposed and de-padded all-atom variables for analysis
                    flat_gt = gt_coords_masked.reshape(gt_coords.shape[0], -1, 3)
                    flat_pred = pred_coords_masked.reshape(pred_coords.shape[0], -1, 3)
                    flat_all_atom_mask = all_atom_mask.reshape(all_atom_mask.shape[0], -1)

                    flat_gt_unpadded = flat_gt[flat_all_atom_mask.bool()]
                    flat_pred_unpadded = flat_pred[flat_all_atom_mask.bool()]
                    flat_gt_unpadded_np = flat_gt_unpadded.cpu().numpy()
                    flat_pred_unpadded_np = flat_pred_unpadded.cpu().numpy()

                    # >>> All-atom RMSD
                    flat_superimposed_pred_aa, rmsd_all = superimpose(flat_gt, flat_pred,
                                                                    flat_all_atom_mask)
                    metrics["rmsd_aa"] = rmsd_all
                    flat_superimposed_pred_aa_unpadded_np = flat_superimposed_pred_aa[
                        flat_all_atom_mask.bool()].cpu().numpy()

                    # >>> Global Metrics (GDC_all, TM score)
                    gdcall_aa = scn_losses.gdc_all(flat_gt_unpadded_np,
                                                flat_superimposed_pred_aa_unpadded_np,
                                                skip_alignment=True,
                                                as_percent=False)
                    tmscore_aa = scn_losses.tm_score(flat_gt_unpadded_np,
                                                    flat_superimposed_pred_aa_unpadded_np,
                                                    skip_alignment=True)
                    tmscore_ca = scn_losses.tm_score(
                        gt_coords_masked_ca[all_atom_mask_ca.bool()].cpu().numpy(),
                        superimposed_pred[all_atom_mask_ca.bool()].cpu().numpy(),
                        skip_alignment=True)

                    # >>> Local Metrics (DRMSD, LDDT, no alignment required)
                    # NOTE-JK: I don't think OpenFold supports all-atom lddt (requires measuring
                    # dists only between atoms in different residues). I use SCN's value instead.
                    drmsd_aa = scn_losses.drmsd(flat_gt_unpadded, flat_pred_unpadded)
                    lddt_aa = scn_losses.lddt_all(
                        flat_gt.reshape(-1, 3) * all_atom_mask.reshape(-1, 1),
                        flat_pred.reshape(-1, 3) * all_atom_mask.reshape(-1, 1),
                        atom_mask=all_atom_mask.reshape(-1).bool(),
                        residue_shape=gt_coords.shape[-1],
                        cutoff=15)
                    lddtquasi_aa = scn_losses.quasi_lddt_all(flat_gt_unpadded,
                                                            flat_pred_unpadded,
                                                            cutoff=15)

                    metrics["gdcall_aa"] = gdcall_aa
                    metrics["tmscore_aa"] = tmscore_aa
                    metrics["tmscore_ca"] = tmscore_ca
                    metrics["drmsd_aa"] = drmsd_aa
                    metrics["lddt_aa"] = lddt_aa
                    metrics["lddtquasi_aa"] = lddtquasi_aa

                    # The below measurements were used to compare sidechainnet's metrics vs OF's
                    # Because they all matched what OpenFold computes, they are not computed again
                    show_scn_metrics = False
                    if show_scn_metrics:
                        # 1. SCN only rmsd (does alignment)
                        alignment_rmsd_scn1 = scn_losses.rmsd(flat_gt_unpadded_np,
                                                            flat_pred_unpadded_np)
                        # 2. SCN assisted rmsd (uses openfold's alignment)
                        alignment_rmsd_scn2 = scn_losses.rmsd(
                            flat_gt_unpadded_np,
                            flat_superimposed_pred_aa[flat_all_atom_mask.bool()].cpu().numpy())
                        # 3. SCN only rmsd_ca (does alignment)
                        alignment_rmsd_scn1_ca = scn_losses.rmsd(
                            gt_coords_masked_ca[all_atom_mask_ca.bool()].cpu().numpy(),
                            pred_coords_masked_ca[all_atom_mask_ca.bool()].cpu().numpy())
                        drmsd_ca_scn = scn_losses.drmsd(
                            gt_coords_masked_ca[all_atom_mask_ca.bool()],
                            pred_coords_masked_ca[all_atom_mask_ca.bool()])
                        lddt_ca_scn = scn_losses.quasi_lddt_all(
                            gt_coords_masked_ca[all_atom_mask_ca.bool()].reshape(-1, 3),
                            pred_coords_masked_ca[all_atom_mask_ca.bool()].reshape(-1, 3),
                            cutoff=15)
                        metrics['rmsdscn1_aa'] = alignment_rmsd_scn1
                        metrics['rmsdscn2_aa'] = alignment_rmsd_scn2
                        metrics['rmsdscn1_ca'] = alignment_rmsd_scn1_ca
                        metrics["drmsd_ca_scn"] = drmsd_ca_scn
                        metrics["lddt_ca_scn"] = lddt_ca_scn

            except ZeroDivisionError:
                pass
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:

        if isinstance(self.trainer.training_type_plugin, DeepSpeedPlugin):
            print("Initializing deepspeed optimizer.")
            optimizer = DeepSpeedCPUAdam(
                self.model.parameters(),
                lr=learning_rate,
                eps=eps
            )
        else:
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
            **self.config.scheduler
        )

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        if(not self.model.template_config.enabled):
            ema["params"] = {k:v for k,v in ema["params"].items() if not "template" in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

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
    config.loss.openmm.use_openmm = args.use_openmm
    config.loss.openmm.add_struct_metrics = args.add_struct_metrics
    config.loss.openmm.weight = args.openmm_weight
    if args.openmm_activation == "sigmoid":
        config.loss.openmm.activation = torch.nn.Sigmoid()
    elif args.openmm_activation == "relu":
        config.loss.openmm.activation = torch.nn.ReLU()
    else:
        config.loss.openmm.activation = None
    config.loss.openmm.write_pdbs = args.write_pdbs
    config.loss.openmm.write_pdbs_every_n_steps = args.write_pdbs_every_n_steps
    config.loss.openmm.pdb_dir = os.path.join(args.output_dir, "pdbs")
    for _subdir in ["true", "pred"]:
        os.makedirs(os.path.join(config.loss.openmm.pdb_dir, _subdir), exist_ok=True)
    config.loss.openmm.use_scn_pdb_names = args.use_scn_pdb_names
    config.loss.openmm.squashed_loss = args.openmm_squashed_loss
    config.loss.openmm.squashed_loss_factor = args.openmm_squashed_loss_factor
    if args.openmm_modified_sigmoid:
        config.loss.openmm.modified_sigmoid = True
        A, B, C, D = map(float, args.openmm_modified_sigmoid.split(","))
        config.loss.openmm.modified_sigmoid_params = (A, B, C, D)


def update_experimental_config(config, args):
    if args.use_lma and args.use_flash_attn:
        raise ValueError("Cannot use both LMA and FlashAttention")
    if args.use_lma:
        config.globals.use_lma = True
        config.globals.use_flash = False
    if args.use_flash_attn:
        config.globals.use_lma = False
        config.globals.use_flash = True
    if args.clear_cache_between_blocks:
        config.model.extra_msa.extra_msa_stack.clear_cache_between_blocks = True
    if args.violation_loss_weight is not None:
        config.loss.violation.weight = args.violation_loss_weight
    if args.exp_resolved_loss_weight is not None:
        config.loss.experimentally_resolved.weight = args.exp_resolved_loss_weight
    if args.scheduler_last_epoch is not None:
        config.scheduler.last_epoch = args.scheduler_last_epoch
    if args.scheduler_max_lr is not None:
        config.scheduler.max_lr = args.scheduler_max_lr
    if args.scheduler_decay_every_n_steps is not None:
        config.scheduler.decay_every_n_steps = args.scheduler_decay_every_n_steps
    if args.use_alphafold_sampling is not None:
        config.data.data_module.use_alphafold_sampling = args.use_alphafold_sampling
    if args.disable_backwards is not None:
        config.model.disable_backwards = args.disable_backwards
    if args.log_to_csv is not None:
        config.logging.log_to_csv = args.log_to_csv
    if args.skip_scnids_file != "":
        with open(args.skip_scnids_file, "r") as f:
            skip_scnids = f.read().splitlines()
        config.data.data_module.skip_scnids = skip_scnids
    if args.grad_clip_val != 0:
        config.model.grad_clip_val = args.grad_clip_val
    if args.scheduler_warmup_no_steps is not None:
        config.scheduler.warmup_no_steps = args.scheduler_warmup_no_steps
    if args.scheduler_base_lr is not None:
        config.scheduler.base_lr = args.scheduler_base_lr
    if args.batch_size != config.data.data_module.data_loaders.batch_size:
        config.data.data_module.data_loaders.batch_size = args.batch_size
        print("Overwriting batch size to", args.batch_size)




def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config(
        args.config_preset,
        train=True,
        low_prec=(str(args.precision) == "16"),
        num_workers=args.num_workers,
    )

    # MOD-JK: Overwrite the config with my experimental args
    update_openmm_config(config, args)
    update_experimental_config(config, args)

    model_module = OpenFoldWrapper(config)
    if args.jax_param_path is not None:
        model_module.model = load_jax_params_into_model(model_module.model,
                                                        args.jax_param_path)
    elif (args.resume_from_ckpt and not args.resume_model_weights_only):
        if (os.path.isdir(args.resume_from_ckpt)):
            last_global_step = get_global_step_from_zero_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt)
            last_global_step = int(sd['global_step'])
        model_module.resume_last_lr_step(last_global_step)
        logging.info("Successfully loaded last lr step...")
    if(args.resume_from_ckpt and args.resume_model_weights_only):
        if(os.path.isdir(args.resume_from_ckpt)):
            sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt)
        # MOD-JK: there is no 'module.' prefix in the state dict; instead, it is missing the expected  `model.` prefix. This applies to initial_training and finetuning.pt files.
        # sd = {k[len("module."):]:v for k,v in sd.items()}
        sd = {"model." + k: v for k, v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")
    if(args.resume_from_jax_params):
        model_module.load_from_jax(args.resume_from_jax_params)
        logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")
    if args.set_lr_step is not None:
        model_module.resume_last_lr_step(args.set_lr_step)
        logging.info(f"Successfully set lr step to {args.set_lr_step}...")
 
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

    loggers = []
    if(args.wandb):
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            notes=args.wandb_notes,
            tags=[tag for tag in args.wandb_tags.split(",") if tag] if args.wandb_tags else None,
            resume=args.wandb_id if args.wandb_id else None,
            group=args.experiment_name if args.experiment_name else "default_group",
            **{"entity": args.wandb_entity},
        )
        # MOD-JK: save config to wandb, log gradients/params
        wdb_logger.experiment.config.update(vars(args), allow_val_change=True)
        loggers.append(wdb_logger)
        print("Experiment dir:", wdb_logger.experiment.dir)
    
    callbacks = []
    if args.checkpoint_every_n_train_steps is not None:
        print("Checkpointing every", args.checkpoint_every_n_train_steps, "train steps.")
        mc = ModelCheckpoint(
            every_n_train_steps=args.checkpoint_every_n_train_steps,
            auto_insert_metric_name=False,
            save_top_k=1,
            monitor="train/loss",
            dirpath=wdb_logger.experiment.dir if args.wandb else None
        )
        callbacks.append(mc)
    elif(args.checkpoint_every_epoch):
        print("Checkpointing every epoch.")
        mc = ModelCheckpoint(
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_top_k=1,
            monitor="train/loss",
            dirpath=wdb_logger.experiment.dir if args.wandb else None
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

    if(not args.do_not_log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)


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

    # MOD-JK Add advanced profiling
    # from pytorch_lightning.profiler import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs03")

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.output_dir,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        # profiler="simple",
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    if args.trainer_mode == "fit" and args.val_data_dir is not None:
        # MOD-JK: validate before fit
        trainer.validate(
            model_module,
            dataloaders=data_module.val_dataloader(), 
            ckpt_path=ckpt_path,
        )
        trainer.fit(
            model_module, 
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    elif args.trainer_mode == "fit":
        trainer.fit(
            model_module, 
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    elif args.trainer_mode == "validate":
        trainer.validate(
            model_module,
            dataloaders=data_module.train_dataloader(), 
            ckpt_path=ckpt_path,
        )
    else:
        raise ValueError(f"Unknown trainer mode: {args.trainer_mode}")


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


def int_or_none_type(int_str):
    if int_str is None or int_str.lower() == "none":
        return None
    else:
        return int(int_str)


def load_jax_params_into_model(param_path, model):
    """Load JAX params into a PyTorch model"""
    model_basename = get_model_basename(param_path)
    model_version = "_".join(model_basename.split("_")[1:])
    import_jax_weights_(model, param_path, version=model_version)
    logging.info(f"Successfully loaded JAX parameters at {path}...")
    return model


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
        "--do_not_log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate. If this flag is not passed, Lr is logged."
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
    parser.add_argument("--num_workers",
                        type=int,
                        default=8,
                        help="Number of workers for data loading.")
    parser.add_argument("--debug",
                        action="store_true",
                        default=False,
                        help="Whether to print debug information from the logger.")
    parser.add_argument("--jax_param_path",
                        type=str,
                        default=None,
                        help="""Path to JAX model parameters.""")
    parser.add_argument("--use_openmm",
                        type=bool_type,
                        default=False,
                        help="Whether to use OpenMM loss.")
    parser.add_argument("--openmm_weight",
                        type=float,
                        default=1.0,
                        help="Weight applied to OpenMM loss.")
    parser.add_argument("--openmm_activation",
                        help="Activation function applied to OpenMM loss. Can be one of "
                        "['sigmoid', 'relu', 'None']. Defaults to 'None'.",
                        choices=["sigmoid", "relu", "None"],
                        default="None")
    parser.add_argument("--add_struct_metrics",
                        action="store_true",
                        default=False,
                        help="Whether to add additional structure metrics to wandb"
                        "including RMSD, GDC, DRMSD, LDDT, etc.")
    parser.add_argument("--write_pdbs",
                        action="store_true",
                        default=False,
                        help="Whether to write pdbs of the predicted structures.")
    parser.add_argument("--write_pdbs_every_n_steps",
                        type=int,
                        default=10,
                        help="Frequency with which to write pdbs of the predicted "
                        "structures.")
    parser.add_argument("--overfit_single_batch",
                        action="store_true",
                        default=False,
                        help="Whether to overfit to the first batch of data.")
    parser.add_argument("--use_scn_pdb_names",
                        "-scnpdb",
                        type=bool_type,
                        default=False,
                        help="Whether to use SidechainNet names for PDB input directory ("
                        "train_data_dir), e.g. 1A9U_1_A.pdb instead of 1a9u.{pdb,mmcif}.")
    parser.add_argument("--wandb_notes", 
                        "-wn",
                        type=str,
                        default=None,
                        help="Notes to add to wandb run.")
    parser.add_argument("--scale_by_length",
                        type=bool_type,
                        default=False,
                        help="Whether to scale the loss by the length of the sequence.")
    parser.add_argument("--force_clipping_val",
                        type=float,
                        default=1e6,
                        help="Value for clipping OpenMM force gradients.")
    parser.add_argument("--use_lma", 
                        type=bool_type,
                        default=False,
                        help="Whether to use low memory attention (mutually exclusive with flash attn).")
    parser.add_argument("--use_flash_attn", 
                        type=bool_type,
                        default=False,
                        help="Whether to use flash attention (mutually exclusive with LMA).")
    parser.add_argument("--clear_cache_between_blocks",
                        type=bool_type,
                        default=False,
                        help="Whether to clear the cache between blocks in extra_msa_stack")
    parser.add_argument("--openmm_squashed_loss",
                        type=bool_type,
                        default=False,
                        help="A modified version of OpenMM loss meant to squash large values."
                        " Loss(e) = e if e < 0 else e*(C/(C+E)).")
    parser.add_argument("--openmm_squashed_loss_factor",
                        type=float,
                        default=100.,
                        help="A modified version of OpenMM loss meant to squash large values."
                        " Loss(e) = e if e < 0 else e*(C/(C+E)).")
    parser.add_argument("--openmm_modified_sigmoid",
                        type=str,
                        default="",
                        help="A modified version of OpenMM loss meant to squash large values."
                             "L(x,a,b,c,d) = (1/a + exp(-(d*x+b)/c))**(-1) - (a-1)."
                             "The argument is a comma-separated list of the parameters a,b,c,d.")
    parser.add_argument("--wandb_tags",
                        type=str,
                        default="",
                        help="Comma-separated list of tags to add to wandb run.")
    parser.add_argument("--violation_loss_weight",
                        type=float,
                        default=None,
                        help="Weight applied to the violation loss.")
    parser.add_argument("--exp_resolved_loss_weight",
                        type=float, 
                        default=None,
                        help="Weight applied to the experimentally resolved loss.")
    parser.add_argument("--set_lr_step",
                        type=int_or_none_type,
                        default=None,
                        help="Step at which to set the learning rate.")
    # parser.add_argument("--scheduler_type",
    #                     type=str,
    #                     default='alphafold',
    #                     help="Type of scheduler to use. Can be one of ['alphafold', 'adam'].")
    parser.add_argument("--scheduler_warmup_no_steps",
                        type=int,
                        default=None,
                        help="Number of steps to warmup the scheduler for.")
    parser.add_argument("--scheduler_base_lr",
                        type=float,
                        default=0,
                        help="Max LR for the scheduler.")
    # parser.add_argument("--scheduler_start_decay_after_n_steps",
    #                     type=int,
    #                     default=None,
    #                     help="Number of steps after which to start decaying the LR.")
    parser.add_argument("--scheduler_last_epoch",
                        type=int_or_none_type,
                        default=-1,
                        help="Step to init the LR scheduler.")
    parser.add_argument("--scheduler_max_lr",
                        type=float,
                        default=.001,
                        help="Max LR for the scheduler.")
    parser.add_argument("--scheduler_decay_every_n_steps",
                        type=int_or_none_type,
                        default=None,
                        help="Number of steps after which to decay the LR.")
    parser.add_argument("--trainer_mode",
                        type=str,
                        default="fit",
                        help="Mode for the trainer. Can be one of ['fit', 'validate']. Defaults to 'fit'.")
    parser.add_argument("--use_alphafold_sampling",
                        type=bool_type,
                        default=True,
                        help="Whether to use the AlphaFold sampling strategy.")
    parser.add_argument("--disable_backwards",
                        type=bool_type,
                        default=False,
                        help="Whether to disable backwards pass.")
    parser.add_argument("--log_to_csv",
                        type=bool_type,
                        default=True,
                        help="Whether to log metrics to a csv file.")
    parser.add_argument("--skip_scnids_file",
                        type=str,
                        default="",
                        help="File containing a list of SidechainNet IDs to skip.")
    parser.add_argument("--grad_clip_val",
                        type=float,
                        default=0,
                        help="Value for clipping gradients.")
    parser.add_argument("--checkpoint_every_n_train_steps",
                        type=int_or_none_type,
                        default=None,
                        help="Number of steps after which to checkpoint the model.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size to use for training.")
    
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

    if args.debug:
        # Set logging level to debug
        logging.basicConfig(level=logging.DEBUG)
        root = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

    main(args)
