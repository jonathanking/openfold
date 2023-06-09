"""This script automates the creation of slurmies for model evaluation."""
import datetime
import pandas as pd
import glob
import os
import argparse

from experiments import Experiment, TrainingExperiment, EvaluationExperiment


def get_checkpoint_path(row):
    """Get the path to the checkpoint to use for an evaluation experiment."""
    if row['exp_name'] == 'initial_training':
        return "openfold/resources/openfold_params/initial_training.pt"
    elif 'finetuning_' in row['exp_name']:
        return f"openfold/resources/openfold_params/{row['exp_name']}.pt"
    if row["exp_suffix"] == "eval_sx":
        # Find the filename with the highest epoch number
        files = os.listdir(f"{row['exp_dir']}/checkpoints")
        files = [f for f in files if f.endswith(".ckpt")]
        files = [f.split("-") for f in files]  # look at epoch and step
        files = [(int(f[0]), int(f[1].split(".")[0])) for f in files]
        files = sorted(files, key=lambda x: (x[0], x[1]))  # find the highest epoch/step
        last_file = files[-1]
        return f"{row['exp_dir']}/checkpoints/{last_file[0]}-{last_file[1]}.ckpt"
    else:
        step = row["exp_suffix"].split("_")[1][1:]
        #  Find the checkpoint with the closest step number
        files = os.listdir(f"{row['exp_dir']}/checkpoints")
        files = [f for f in files if f.endswith(".ckpt")]
        files = [f.split("-") for f in files]
        files = [(int(f[0]), int(f[1].split(".")[0])) for f in files]
        # Get the closest step number
        closest_step = min(files, key=lambda x: abs(x[1] - int(step)))
        # Return the path to the closest checkpoint
        return f"{row['exp_dir']}/checkpoints/{closest_step[0]}-{closest_step[1]}.ckpt"


def create_eval_job_df(exp_dir, eval_jobs, location):
    """Create a dataframe with the information needed to create evaluation jobs."""

    exp_dir = pd.read_csv(exp_dir)
    eval_jobs = pd.read_csv(eval_jobs)
    # Remove rows in eval_jobs where the column 'eval_me' is False
    eval_jobs = eval_jobs[eval_jobs["eval_me"] == True]
    df = eval_jobs.merge(exp_dir, on="exp_name", how="left")


    df['exp_suffix'].fillna('eval_s!', inplace=True)
    df["new_exp_name"] = df["exp_name"] + "-" + df["exp_suffix"]
    df["new_exp_name"] = df["new_exp_name"].apply(lambda x: x.replace("-", "_"))

    df = df[df["location"] == location]

    # Find the actual experiment directory
    df["exp_dir"] = df["wandb_id"].apply(
        lambda x: f"out/experiments/*/finetune-openfold-02/{x}")
    def get_dir(x):
        try:
            return glob.glob(x)[0]
        except IndexError:
            return None
    try:
        df["exp_dir"] = df["exp_dir"].apply(get_dir)
    except IndexError:
        print("No experiment directory found. Check the wandb_id.")
        print(df["exp_dir"].values)
        print("Exiting...")
        exit()

    df["checkpoint_path"] = df.apply(get_checkpoint_path, axis=1)

    # get the step from the checkpoint path
    def get_step(row):
        if row['exp_name'] == 'initial_training' or 'finetuning_' in row['exp_name']:
            return 0
        else:
            return int(row['checkpoint_path'].split('/')[-1].split('-')[1].split('.')[0])

    df['step'] = df.apply(get_step, axis=1)

    # replace the eval_sx with the actual step number
    df['new_exp_name'] = df['new_exp_name'].apply(lambda x: x.replace(
        'eval_sx', 'eval_s' + str(df[df['new_exp_name'] == x]['step'].values[0])))

    return df


def create_slurm_eval_experiments(df):
    """Return a list of EvaluataionExperiment objects."""
    # Create EvaluationExperiment objects from the dataframe
    eval_exps = []
    for i, row in df.iterrows():
        step = row["exp_suffix"].split("_")[1][1:]
        if step == "x":
            step = row['step']
        eval_exp = EvaluationExperiment(
            exp_name=row["new_exp_name"],
            wandb_id=row["wandb_id"],
            location=row["location"],
            notes=f"Evaluating {row['exp_name']} at step {step}.",
            exp_suffix=row["exp_suffix"],
            checkpoint_path=row["checkpoint_path"],
        )
        eval_exps.append(eval_exp)

    return eval_exps


def main(args):
    df = create_eval_job_df(args.exp_directory, args.eval_job_csv, args.location)

    # Get slurm scripts based on the dataframe
    slurm_eval_experiments = create_slurm_eval_experiments(df)

    # Write slurm scripts to file in jk_research/evaluations/{6digitdate}/subjobs
    # First create the directory if it doesn't exist
    date = datetime.datetime.now().strftime("%y%m%d")
    slurm_dir = f"jk_research/evaluations/{date}/subjobs"
    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)

    for i, slurm_eval_exp in enumerate(slurm_eval_experiments):
        slurm_eval_exp.write_slurm_script(f"eval{i}", slurm_dir)
        slurm_eval_exp.run_slurm_script()

    # Run slurm scripts

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_directory',
                        type=str,
                        default="jk_research/evaluations/experiment_directory.csv",
                        help='Path to the experiment directory.')
    parser.add_argument('--eval_job_csv',
                        type=str,
                        default="jk_research/evaluations/eval_jobs.csv",
                        help='Path to the csv file containing the evaluation jobs.')
    parser.add_argument('--location',
                        type=str,
                        help='Location of the experiment.',
                        default="g019")
    args = parser.parse_args()

    main(args)