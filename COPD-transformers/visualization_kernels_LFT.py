# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.visualization.plots_kernels_LFT import ConfusionKernels_LFT, OneOffKernels_LFT

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="")
parser.add_argument("--input", type=str, help="E,C,T,EC,CT,ET,ETC",default="CT")
parser.add_argument("--output", type=str, help="EMPH, TRAJ or COPD",default="TRAJ")
parser.add_argument("--add", type=int, help="1: age, 2: age+gender, 3: age+gender+packs",default=0)
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="modular")

args = parser.parse_args()

main_path = args.path
project_name = args.project_name
how = args.how
vars_add = args.add
output_type = args.output
input_type = args.input
wrapping_mode = args.wrap

save_path = main_path+project_name+'-results/figures/kernels-LFT'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)
                
for i in range(0,6):
    kernel = i+1
    for num_slices in [20]:
        print(f"\nResults LFT {output_type} using {input_type} with {how} adding {vars_add} with {wrapping_mode} ---- {num_slices} k{kernel}")
        conf_mat = ConfusionKernels_LFT(main_path,
                    num_slices,
                    how,
                    project_name,
                    output_type,
                    input_type,
                    vars_add,
                    kernel,
                    wrapping_mode)
        one_off = OneOffKernels_LFT(main_path,
                    num_slices,
                    how,
                    project_name,
                    output_type,
                    input_type,
                    vars_add,
                    kernel,
                    wrapping_mode)

print('')       
print('Finished')