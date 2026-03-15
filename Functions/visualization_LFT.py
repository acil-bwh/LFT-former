# ··························································
# ············ Applied Chest Imaging Lab, 2026 ·············
# ·········· @acil-bwh | @queraltmartinsaladich ············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.visualization.plots_LFT import Confusion_LFT, OneOff_LFT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")
parser.add_argument("--add", type=int, help="1: age, 2: age+gender, 3: age+gender+packs",default=0)
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="modular")

args = parser.parse_args()

main_path = args.path
project_name = args.project_name
vars_add = args.add
wrapping_mode = args.wrap

save_path = main_path+project_name+'-results/figures/LFT'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"\nResults LFT adding {vars_add} with {wrapping_mode} ---- ")
conf_mat = Confusion_LFT(main_path,
                project_name,
                vars_add,
                wrapping_mode)
one_off = OneOff_LFT(main_path,
                project_name,
                vars_add,
                wrapping_mode)