# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from Utils.training.training_model_mOutViT import Trainer_mOutViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path", required=True)

parser.add_argument("--cuda", type=int, help="cuda node id", default=0)
parser.add_argument("--epochs", type=int, help="epochs",default=200)
parser.add_argument("--batch", type=int, help="batch size: 64 OR 32",default=64)
parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights')
parser.add_argument('--precheck', action='store_true', help='Load from checkpoint')
parser.add_argument("--project_name", type=str, help="project dataset name, e.g. COPDGene",default="COPDGene")
parser.add_argument("--slices", type=int, help="9 or 20 slices",default=20)
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="concat")
parser.add_argument("--lr", type=float, help="learning rate",default=1e-4)
parser.add_argument("--decay", type=float, help="weight decay",default=1e-5)
parser.add_argument("--gamma", type=float, help="gamma",default=0.9)
parser.add_argument("--step", type=int, help="step",default=5)
parser.add_argument("--input", type=str, help="E,C,T,EC,CT,ET,ETC",default="CT")
parser.add_argument("--output", nargs='+', help="any list of: E,C,DW,VNB-LD,VA-LD,FEV1,Eos,pE",default=["E"])

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
epochs = args.epochs
batch_size = args.batch
load_pretrained = args.pretrained
load_from_checkpoint = args.precheck
how = args.how
project_name = args.project_name
slices = args.slices
weight_decay = args.decay
lr = args.lr
gamma = args.gamma
step = args.step
input_type = args.input
output_type = args.output

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

which_outputs = [output_mapping_dict[code] for code in output_type]

filtered_dict_out = filter_dict_by_keys(loaded_dict_out, which_outputs)
filtered_task_dict = filter_dict_by_keys(loaded_task_dict, which_outputs)

print("\n--- Filtered multi-output keys: ---")
print(filtered_dict_out.keys()) 

if load_pretrained and load_from_checkpoint:
    raise ValueError("You cannot set both --pretrained and --precheck at the same time.")
elif not load_pretrained and not load_from_checkpoint:
    print("\nNeither pretrained nor checkpoint selected — starting from scratch.")

predicting = Trainer_mOutViT(main_path,
            num_slices=slices,
            cuda_id=cuda_id,
            project_name=project_name,
            epochs=epochs,
            batch_size=batch_size,
            dict_out=filtered_dict_out,
            task_dict=filtered_task_dict,
            input_type=input_type,
            how=how,
            load_pretrained=load_pretrained,
            load_from_checkpoint=load_from_checkpoint,
            lr=lr,
            gamma=gamma,
            weight_decay=weight_decay,
            step=step)

print('Finished')