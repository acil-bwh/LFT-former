# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.inference.get_probabilities_augViT import Probabilities_augViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--num_slices", help="num_slices size: 1, 9 or 20")
parser.add_argument("--id", help="patient id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")
parser.add_argument("--model", type=int, help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
num_slices = args.num_slices
patient_sid = args.id
project_name = args.project_name
model_id = args.model

save_path = main_path+project_name+'-results/models'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

for num_slices in [20]:
    predicting = Probabilities_augViT(
        main_path,
        model_id,
        num_slices,
        cuda_id,
        patient_sid,
        project_name)

print('Finished')