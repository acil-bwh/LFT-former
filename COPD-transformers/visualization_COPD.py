# ············· QUERALT MARTÍN-SALADICH, 2024 ··············
# ···················· BWH + UPF + VHIR ····················
# ··························································

from Utils.visualization.plots_COPD import Confusion_COPD
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--model", help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)
parser.add_argument("--output", help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")

args = parser.parse_args()

main_path = args.path
model = args.model
output = args.output
project_name = args.project_name

save_path = main_path+project_name+'-results/figures'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

for mode_type in ["original","five_class","binary"]:
    print(f"Results input augViT i{model} o{output} - { mode_type}")
    conf_mat = Confusion_COPD(main_path,
                    model_id=int(model),
                    output_id=int(output),
                    mode=mode_type)