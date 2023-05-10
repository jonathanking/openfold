"""This script contains some python wrapper functions to evaluate model quality."""

import subprocess
import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm



def compute_cadaa_score(model_file, target_file, mode='AA'):
    """Use the commandline tool voronota-cadscore to compute the CAD (all-atom) score."""
    if mode != 'AA':
        raise NotImplementedError("Only AA mode is implemented.")
    cmd = f"voronota-cadscore -m {model_file} -t {target_file} --contacts-query-by-code {mode}"
    # print(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True)
    # print(output)
    output = output.stdout.decode("utf-8").split(" ")
    output_labels = ['t_filepath', 'm_filepath', 'query_code', 'num_res', 'global_score', 't_totalarea', 'm_totalarea']
    assert len(output) == len(output_labels), 'Output of voronota-cadscore is not as expected. ' + str(output)
    output = dict(zip(output_labels, output))
    return float(output['global_score'])


def compute_all_cadaa_scores(model_dir_dir, mode='AA'):
    """Compute the CAD (all-atom) score for all models in a directory."""
    eval_model_scores = {}
    eval_model_names = os.listdir(model_dir_dir)
    for model_name in eval_model_names:
        eval_model_scores[model_name] = {}

    # Get CAD-AA scores for the models
    for eval_model_name in eval_model_names[:2]:
        model_dir = os.path.join(model_dir_dir, eval_model_name)
        model_pdbs = glob.glob(os.path.join(model_dir, 'pdbs', 'pred', '*.pdb'))

        for model_pdb in model_pdbs[:7]:
            target_pdb = model_pdb.replace('pred', 'true')
            model_pdb_name = os.path.basename(model_pdb).split('.')[0][10:]
            print(model_pdb_name)
            eval_model_scores[eval_model_name][model_pdb_name] = compute_cadaa_score(model_pdb, target_pdb, mode=mode)
    return eval_model_scores


def compute_all_cadaa_scores_parallel(model_dir_dir, mode='AA', max_workers=None):
    """Compute the CAD (all-atom) score for all models in a directory."""
    eval_model_scores = {}
    eval_model_names = os.listdir(model_dir_dir)
    for model_name in eval_model_names:
        eval_model_scores[model_name] = {}

    def process_model_pdb(eval_model_name, model_pdb):
        target_pdb = model_pdb.replace('pred', 'true')
        model_pdb_name = os.path.basename(model_pdb).split('.')[0][10:]
        score = compute_cadaa_score(model_pdb, target_pdb, mode=mode)
        return eval_model_name, model_pdb_name, score

    tasks = []
    for eval_model_name in eval_model_names:
        model_dir = os.path.join(model_dir_dir, eval_model_name)
        model_pdbs = glob.glob(os.path.join(model_dir, 'pdbs', 'pred', '*.pdb'))
        for model_pdb in model_pdbs:
            tasks.append((eval_model_name, model_pdb))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(lambda args: process_model_pdb(*args), tasks), total=len(tasks)))

    for eval_model_name, model_pdb_name, score in results:
        eval_model_scores[eval_model_name][model_pdb_name] = score

    return eval_model_scores



def main(model_dir_dir, cad_mode, output_csv):
    """Main function."""
    
    eval_model_scores = compute_all_cadaa_scores_parallel(model_dir_dir, mode=cad_mode)
    eval_model_scores_df = pd.DataFrame(eval_model_scores)
    eval_model_scores_df.index.name = 'protein_name'
    eval_model_scores_df.to_csv('cadaa_scores.csv')
    eval_model_scores_df.to_csv(output_csv)


if __name__ == "__main__":
    # use argparse to get the model_dir_dir, cad_mode, and output_csv
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_dir_dir", type=str, help="Directory containing the model directories.", default="/net/pulsar/home/koes/jok120/openfold/out/evaluation/230507")
    argparser.add_argument("cad_mode", type=str, help="CAD mode to use.", default="AA")
    argparser.add_argument("output_csv", type=str, help="Output csv file.", default="cadaa_scores.csv")
    args = argparser.parse_args()
    main(args.model_dir_dir, args.cad_mode, args.output_csv)