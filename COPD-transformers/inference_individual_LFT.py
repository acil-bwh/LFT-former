# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2026 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.inference.get_probabilities_LFT import Probabilities_LFT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--id", help="patient id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")
parser.add_argument("--add", type=int, help="0: none, 1: age, 2: age+gender, 3: age+gender+packs, 4:age+gender+packs+emph...",default=0)
parser.add_argument("--wrap", type=str, help="modular, gate, gatt, cross, multi, along, dot or concat",default="gatt")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
patient_sid = args.id
project_name = args.project_name
vars_add = args.add
wrapping_mode = args.wrap

save_path = main_path+project_name+'-results/models'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

predicting = Probabilities_LFT(
    main_path,
    cuda_id,
    patient_sid,
    project_name,
    vars_add,
    wrapping_mode)

print('Finished')