# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.inference.predictions_RegionViT import Predictor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--model", type=int, help="1 for emphysema, 2 for trajectories, 3 for COPD, 4 for air trapping",default=1)
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPD",default="COPDGene")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
model = args.model
project_name = args.project_name

save_path = main_path+project_name+'-results/models'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

for batch_size in [9,20]:
    predicting = Predictor(main_path=main_path,
                    model_id=model,
                    batch_size=batch_size,
                    cuda_id=int(cuda_id),
                    project_name=str(project_name))
    
print('Finished')