# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.inference.get_probabilities_LFT import Probabilities_LFT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--num_slices", help="num_slices size: 1, 9 or 20")
parser.add_argument("--id", help="patient id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="")
parser.add_argument("--input", type=str, help="E,C,T,EC,CT,ET,ETC",default="T")
parser.add_argument("--output", type=str, help="EMPH, TRAJ or COPD",default="TRAJ")
parser.add_argument("--add", type=int, help="1: age, 2: age+gender, 3: age+gender+packs",default=0)
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="gatt")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
num_slices = args.num_slices
patient_sid = args.id
project_name = args.project_name
how = args.how
vars_add = args.add
output_type = args.output
input_type = args.input
wrapping_mode = args.wrap

save_path = main_path+project_name+'-results/models'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

for num_slices in [20]:
    predicting = Probabilities_LFT(
        main_path,
        num_slices,
        cuda_id,
        patient_sid,
        how,
        project_name,
        output_type,
        input_type,
        vars_add,
        wrapping_mode)

print('Finished')