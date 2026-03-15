# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from Utils.training.training_model_mvLFT import Trainer_mvLFT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path", required=True)

parser.add_argument("--cuda", type=int, help="cuda node id", default=0)
parser.add_argument("--epochs", type=int, help="epochs",default=500)
parser.add_argument("--batch", type=int, help="batch size: 64 OR 32",default=64)
parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights')
parser.add_argument('--precheck', action='store_true', help='Load from checkpoint')
parser.add_argument("--project_name", type=str, help="project dataset name, e.g. COPDGene",default="COPDGene")
parser.add_argument("--slices", type=int, help="9 or 20 slices",default=9)
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="")
parser.add_argument("--view", type=str, help="how are views fused -- concat/mean/max",default="concat")
parser.add_argument("--lr", type=float, help="learning rate",default=1e-6)
parser.add_argument("--decay", type=float, help="weight decay",default=1e-9)
parser.add_argument("--gamma", type=float, help="gamma",default=0.9)
parser.add_argument("--step", type=int, help="step",default=10)
parser.add_argument("--optzr", type=str, help="optimizer: adam or sgd",default="adam")
parser.add_argument("--schdr", type=str, help="scheduler: step or plat",default="step")
parser.add_argument("--input", type=str, help="E,C,T,EC,CT,ET,ETC",default="CT")
parser.add_argument("--output", type=str, help="EMPH, TRAJ or COPD",default="TRAJ")
parser.add_argument("--add", type=int, help="0: none, 1: age, 2: age+gender, 3: age+gender+packs, 4:age+gender+packs+emph",default=0)
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="modular")

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
optzr = args.optzr
schdr = args.schdr
output_type = args.output
input_type = args.input
vars_add = args.add
wrapping_mode = args.wrap
how_view = args.view

if load_pretrained and load_from_checkpoint:
    raise ValueError("You cannot set both --pretrained and --precheck at the same time.")
elif not load_pretrained and not load_from_checkpoint:
    print("Neither pretrained nor checkpoint selected — starting from scratch.")

predicting = Trainer_mvLFT(main_path,
               slices,
               cuda_id,
               project_name,
               epochs,
               batch_size,
               load_pretrained,
               how,
               how_view,
               load_from_checkpoint,
               lr,
               gamma,
               step,
               optzr,
               schdr,
               weight_decay,
               output_type,
               input_type,
               vars_add,
               wrapping_mode)

print('Finished')