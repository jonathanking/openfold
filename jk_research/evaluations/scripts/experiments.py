"""Experiment classes for OpenFold finetuning research."""


import datetime
import os
import subprocess


class Experiment:
    def __init__(self,
                 exp_name,
                 wandb_id,
                 location,
                 notes,
                 slurm_template="jk_research/evaluations/scripts/subjob_skeleton.slurm"):
        assert "-" not in exp_name, "Experiment name cannot contain dashes."

        self.exp_name = exp_name
        self.wandb_id = wandb_id
        self.location = location
        self.notes = notes
        self.mode = "unk"
        self.slurm_template_path = slurm_template

        if self.location == "g019":
            self.code_base_path = "/net/pulsar/home/koes/jok120/openfold"
        elif self.location == "crc":
            self.code_base_path = "/ihome/dkoes/jok120/openfold"
        else:
            raise ValueError(f"Invalid location: {self.location}")

        self._set_exp_defaults()

    @property
    def eval_summary_dir(self):
        return f"{self.code_base_path}/out/evaluations/{self.exp_name}"

    def _set_exp_defaults(self):
        self.nodes = 1
        self.cluster = "gpu" if self.location == "crc" else None
        self.nodelist = "g019" if self.location == "g019" else None
        self.output_log = f"{self.code_base_path}/out/%A_%6a.out"

        self.valmin_structures_dir = "data/validation/cameo/20220116/minimized/data_dir"
        self.valmin_alignments_dir = "data/validation/cameo/20220116/minimized/alignments"
        self.testmin_structures_dir = "data/test/cameo/20230103/minimized/data_dir"
        self.testmin_alignments_dir = "data/test/cameo/20230103/minimized/alignments"

        # Assign variables much like was done above for the following
        self.train_structures_dir = "data/train_structs/scnmin_structs/"
        self.alignments_dir = "data/alignments/scnmin_alignments/"
        self.template_structures_dir = "data/template_structs/roda_pdbs_snapshotted_flattened_do_not_overwrite/"
        self.train_cache = "data/caches/chain_data_cache_rodasnapshot_clustered.json"
        self.template_cache = "data/caches/mmcif_cache_rodasnapshot.json"
        self.resume_model_weights_only = True if "initial_training" in self.exp_name else False

        self.module_loads = "module load cuda/11.3.0\nmodule load gcc/8.2.0" if self.location == "crc" else "module load cuda/11.5"

    def get_slurm_body(self):
        with open(self.slurm_template_path, "r") as f:
            slurm_str = f.read()
        return slurm_str

    def get_slurm_preamble(self,
                           job_name,
                           nodes=None,
                           gpus=None,
                           ntasks=None,
                           partition=None,
                           time=None,
                           output=None,
                           qos=None,
                           cluster=None,
                           nodelist=None):
        nodes = nodes if nodes is not None else self.nodes
        gpus = gpus if gpus is not None else self.gpus
        ntasks = ntasks if ntasks is not None else self.ntasks
        partition = partition if partition is not None else self.partition
        time = time if time is not None else self.time
        output = output if output is not None else self.output_log
        qos = qos if qos is not None else self.qos
        cluster = cluster if cluster is not None else self.cluster
        nodelist = nodelist if nodelist is not None else self.nodelist

        preamble = f"""#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --output="{output}"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jok120@pitt.edu"""

        if qos is not None:
            preamble += f"\n#SBATCH --qos={qos}"
        if cluster is not None:
            preamble += f"\n#SBATCH --cluster={cluster}"
        if nodelist is not None:
            preamble += f"\n#SBATCH --nodelist={nodelist}"

        return preamble

    def get_slurm_script_str(self, job_name):
        preamble = self.get_slurm_preamble(job_name)
        body = self.get_slurm_body()
        date = datetime.datetime.now().strftime("%y%m%d")
        body = body.format(
            preamble=preamble,
            EXPERIMENT_NAME=self.exp_name,
            OUTDIR=f"out/evaluation/{date}/{self.exp_name}",
            CHECKPOINT=self.checkpoint_path,
            VALIDATION_STRUCTURES_DIR=self.valmin_structures_dir,
            VALIDATION_ALIGNMENTS_DIR=self.valmin_alignments_dir,
            TEST_STRUCTURES_DIR=self.testmin_structures_dir,
            TEST_ALIGNMENTS_DIR=self.testmin_alignments_dir,
            TRAIN_STRUCTURES_DIR=self.train_structures_dir,
            ALIGNMENTS_DIR=self.alignments_dir,
            TEMPLATE_STRUCTURES_DIR=self.template_structures_dir,
            TRAIN_CACHE=self.train_cache,
            TEMPLATE_CACHE=self.template_cache,
            NOTES=self.notes,
            module_loads=self.module_loads,
            resume_model_weights_only=self.resume_model_weights_only,
        )
        return body

    def write_slurm_script(self, job_name, outdir):
        script_str = self.get_slurm_script_str(job_name)
        script_path = os.path.join(outdir, f"{self.exp_name}.slurm")
        with open(script_path, "w") as f:
            f.write(script_str)
        self.script_path = script_path
    
    def run_slurm_script(self):
        cmd = f"sbatch {self.script_path}"
        p = subprocess.Popen(cmd, cwd=self.code_base_path, shell=True)
        p.wait()
        print(cmd)
        


class TrainingExperiment(Experiment):
    def __init__(self, exp_name, wandb_id, location, notes):
        super().__init__(exp_name, wandb_id, location, notes)
        self._set_defaults()

    def _set_defaults(self):
        self.gpus = 4
        self.ntasks = 64
        self.partition = "a100_nvlink" if self.location == "crc" else "dept_gpu"
        self.time = "6-00:00:00" if self.location == "crc" else "14-00:00:00"
        self.qos = "long" if self.location == "crc" else None


class EvaluationExperiment(Experiment):
    def __init__(self, exp_name, wandb_id, location, notes, exp_suffix, checkpoint_path):
        super().__init__(exp_name, wandb_id, location, notes)
        self.exp_suffix = exp_suffix
        self.checkpoint_path = checkpoint_path
        self._set_defaults()

    def _set_defaults(self):
        self.nodes = 1
        self.gpus = 1
        self.ntasks = 4
        self.partition = "a100" if self.location == "crc" else "dept_gpu"
        self.time = "03:00:00" if self.location == "crc" else "6-00:00:00"
        self.qos = "short" if self.location == "crc" else None
