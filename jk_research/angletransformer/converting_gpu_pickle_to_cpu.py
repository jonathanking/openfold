# coding: utf-8
import pickle
import torch
with open("out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates0_train.pkl", "rb") as f:
    d = pickle.load(f)
    
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
with open("out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates0_train.pkl", "rb") as f:
    contents = CPU_Unpickler(f).load()
    
import io
with open("out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates0_train.pkl", "rb") as f:
    contents = CPU_Unpickler(f).load()
    
contents.keys()
contents['AT'].keys()
len(contents['AT']['s'])
contents['AT']['s'][0]
contents['AT']['s'][0].device
with open("out/experiments/angletransformer-make-caches-10-65kpt1/cpus/angle_transformer_intermediates0_train_debug.pkl", "wb") as f:
    pickle.dump(contents, f)
    
get_ipython().run_line_magic('mkdir', 'cpus')
get_ipython().system('ls')
with open("out/experiments/angletransformer-make-caches-10-65kpt1/cpus/angle_transformer_intermediates0_train_debug.pkl", "wb") as f:
    pickle.dump(contents, f)
    
def process(fn):
    fn_out = fn.replace("angle_transformer_inter", "cpus/angle_transformer_inter")
    cpu_contents = CPU_Unpickler(f).load()
def process(fn):
    fn_out = fn.replace("angle_transformer_inter", "cpus/angle_transformer_inter")
    with open(fn, "rb") as f:
        cpu_contents = CPU_Unpickler(f).load()
    with open(fn_out, "wb") as f_out:
        pickle.dump(cpu_contents, f_out)
        
def process(fn):
    fn_out = fn.replace("angle_transformer_inter", "cpus/angle_transformer_inter")
    print("Reading...", end=" ")
    with open(fn, "rb") as f:
        cpu_contents = CPU_Unpickler(f).load()
    print("done.\nWriting...", end= " ")
    with open(fn_out, "wb") as f_out:
        pickle.dump(cpu_contents, f_out)
    print("done.")
    
process("out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates1_train.pkl)
process("out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates1_train.pkl")
def process(fn):
    fn_out = fn.replace("angle_transformer_inter", "cpus/angle_transformer_inter")
    print("Reading...", end=" ", flush=True)
    with open(fn, "rb") as f:
        cpu_contents = CPU_Unpickler(f).load()
    print("done.\nWriting...", end= " ", flush=True)
    with open(fn_out, "wb") as f_out:
        pickle.dump(cpu_contents, f_out)
    print("done.", flush=True)
    
process("out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates1_train.pkl")
files = []
files = [f"out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates{i}_train.pkl" for i in range(2,5)]
files
files += [f"out/experiments/angletransformer-make-caches-10-65kpt2/angle_transformer_intermediates{i}_train.pkl" for i in range(0,5)]
files
files += [f"out/experiments/angletransformer-make-caches-10-65kpt2/angle_transformer_intermediates{i}_val.pkl" for i in range(0,5)]
files
files = [f if "4" not in f for f in files]
files = [f for f in files if "4" not in f]
files
files += [f"out/experiments/angletransformer-make-caches-10-65kpt1/angle_transformer_intermediates{i}_val.pkl" for i in range(0,4)]
files
get_ipython().system('pwd')
get_ipython().run_line_magic('notebook', '-e converting_gpu_pickle_to_cpu.ipynb')
get_ipython().run_line_magic('notebook', 'converting_gpu_pickle_to_cpu.ipynb')
