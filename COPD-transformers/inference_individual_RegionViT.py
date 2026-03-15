# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from Utils.inference.get_probabilities_RegionViT import Probabilities
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--model", help="model: 1 (emphysema), 2 (trajectories) or 3 (COPD)")
parser.add_argument("--id", help="patient id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPD")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
model_id = args.model
id_num = args.id
project_name = args.project_name

save_path = main_path+project_name+'-results/models'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

for batch_size in [1,9,20]:
        predicting = Probabilities(main_path,
                model_id=int(model_id),
                batch_size=int(batch_size),
                cuda_id=int(cuda_id),
                id=id_num,
                project_name=project_name)
        print(" ")

print('Finished')