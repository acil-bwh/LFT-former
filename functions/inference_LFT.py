# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.inference.predictions_LFT import Predictor_LFT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPD")
parser.add_argument("--add", type=int, help="0: none, 1: age, 2: age+gender, 3: age+gender+packs, 4:age+gender+packs+emph...",default=0)
parser.add_argument("--wrap", type=str, help="modular, gate, gatt, cross, multi, along, dot or concat",default="gatt")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
project_name = args.project_name
vars_add = args.add
wrapping_mode = args.wrap

for num_slices in [20]:
    print(f"\nPredicting LFT adding {vars_add} with {wrapping_mode} ---- {num_slices}")
    predicting = Predictor_LFT(main_path,
                                num_slices,
                                cuda_id,
                                project_name,
                                vars_add,
                                wrapping_mode)
    
print('Finished')