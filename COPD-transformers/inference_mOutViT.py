# ············· QUERALT MARTÍN-SALADICH, 2024 ··············
# ···················· BWH + UPF + VHIR ····················
# ··························································

from Utils.inference.predictions_mOutViT import Predictor_mOutViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPD",default="COPDGene")
parser.add_argument("--input", type=str, help="E,C,T,EC,CT,ET,ETC",default="CT")
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="concat")
parser.add_argument("--output", nargs='+', help="any list of: E,C,DW,VNB-LD,VA-LD,FEV1,Eos,pE",default=["E"])

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
project_name = args.project_name
input_type = args.input
output_type = args.output
how = args.how

import json

def load_dict_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_dict_by_keys(original_dict, keys_to_keep):
    filtered_dict = {k: v for k, v in original_dict.items() if k in keys_to_keep}
    return filtered_dict

source_files = f"{main_path}/{project_name}-files"
loaded_dict_out = load_dict_from_json(f"{source_files}/dict_out_config_class.json")
loaded_task_dict = load_dict_from_json(f"{source_files}/task_dict_config_class.json")

available_outputs = ["emph_cat_P1", "finalgold_visit_P1",
                     "distwalked_P1","lung_density_vnb_P1",
                     "FEV1pp_post_P1",
                     "eosinphl_P2","pctEmph_Thirona_P1",
                     "Pi10_Thirona_P1",
                     "ChangeFEV1pp_P1P2","ChangeVA_LD_P1P2",
                     "Changedistwalked_P1P2","ChangeVNB_LD_P1P2"]

calling_outputs = ["E","C","DW","VNB-LD","FEV1","Eos","pE","Pi10","dFEV1","dVA-LD","dDW","dVNB-LD"]

output_mapping_dict = dict(zip(calling_outputs, available_outputs))
if isinstance(output_type, str):
    output_type = [output_type]

try:
    which_outputs = [output_mapping_dict[code] for code in output_type]
except KeyError as e:
    print(f"\n[!] Error: The output code '{e.args[0]}' is not recognized.")
    print(f"Available codes are: {list(output_mapping_dict.keys())}")
    exit(1)

which_outputs = [output_mapping_dict[code] for code in output_type]

filtered_dict_out = filter_dict_by_keys(loaded_dict_out, which_outputs)
filtered_task_dict = filter_dict_by_keys(loaded_task_dict, which_outputs)

print("\n--- Filtered multi-output keys: ---")
print(filtered_dict_out.keys()) 



how = "concat"
num_slices = 20
predicting = Predictor_mOutViT(main_path,
                num_slices,
                cuda_id,
                filtered_dict_out,
                filtered_task_dict,
                how,
                project_name,
                input_type)
    
print('Finished')