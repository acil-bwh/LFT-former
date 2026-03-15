# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from Utils.inference.get_probabilities_mOutViT import Probabilities_mOutViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--num_slices", help="num_slices size: 1, 9 or 20")
parser.add_argument("--id", help="patient id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPD")
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
num_slices = args.num_slices
id_num = args.id
project_name = args.project_name
how = args.how

save_path = main_path+project_name+'-results/models'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

dict_out = {
"EMPH": [0, 1, 2, 3],          # Classification task 1 EMPH: 4 classes
"TRAJ": [1, 2, 3, 4, 5, 6],    # Classification task 2 TRAJ: 6 classes
"COPD": [-1, 0, 1, 2, 3, 4]}   # Classification task 3 COPD: 6 classes

for batch_size in [9,20]:
    if batch_size == 9:
        how = "weighted"
    else:
        how = "mean"    
    
    predicting = Probabilities_mOutViT(main_path=main_path,
                    num_slices=batch_size,
                    cuda_id=int(cuda_id),
                    dict_out=dict_out,
                    how=how,
                    project_name=project_name,
                    patient_id=id_num)

print('Finished')