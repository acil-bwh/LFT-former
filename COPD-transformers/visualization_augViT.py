# ············· QUERALT MARTÍN-SALADICH, 2024 ··············
# ···················· BWH + UPF + VHIR ····················
# ··························································

from Utils.visualization.plots_augViT import Confusion_augViT, OneOff_augViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--model", help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)
parser.add_argument("--output", help="1 for emphysema, 2 for trajectories, 3 for COPD, 5 for COPD-5, 6 for COPD-2",default=3)
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

for batch_size in [20]:
    print(f"Results input augViT i{model} o{output} ---- {batch_size}")
    conf_mat = Confusion_augViT(main_path,
                    model_id=int(model),
                    batch_size=batch_size,
                    output_id=int(output))
    one_off = OneOff_augViT(main_path=main_path,
                        model_id=int(model),
                        batch_size=batch_size,
                        output_id=int(output))