# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from Utils.visualization.plots_RegionViT import Confusion,OneOff
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--model", help="1 for emphysema, 2 for trajectories, 3 for COPD, 4 for gas trapping")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPD")
args = parser.parse_args()

main_path = args.path
model = args.model
project_name = args.project_name

save_path = main_path+project_name+'-results/figures/RegionViT'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

for batch_size in [1,9,20]:
    print("Results ---- ",str(batch_size),"-slices")
    conf_mat = Confusion(main_path,
                    model_id=int(model),
                    batch_size=batch_size,
                    project_name=project_name)
    one_off = OneOff(main_path=main_path,
                        model_id=int(model),
                        batch_size=batch_size,
                        project_name=project_name)