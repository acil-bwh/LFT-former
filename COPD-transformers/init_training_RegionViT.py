# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
from Utils.training.training_model_RegionViT import Trainer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, help="main path", required=True)
parser.add_argument("--cuda", type=int, help="cuda node id", default=0)
parser.add_argument("--epochs", type=int, help="epochs",default=500)
parser.add_argument("--batch", type=int, help="batch size: 64 OR 32",default=64)
parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights')
parser.add_argument('--precheck', action='store_true', help='Load from checkpoint')
parser.add_argument("--model", type=int, help="1 for emphysema, 2 for trajectories, 3 for COPD",default=2)
parser.add_argument("--project_name", type=str, help="project dataset name, e.g. COPDGene",default="COPDGene")
parser.add_argument("--lr", type=float, help="learning rate",default=1e-4)
parser.add_argument("--decay", type=float, help="weight decay",default=1e-7)
parser.add_argument("--gamma", type=float, help="gamma",default=0.9)
parser.add_argument("--step", type=int, help="step",default=10)
parser.add_argument("--axis", type=str, help="coronal or axial",default="axial")
args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
epochs = args.epochs
batch_size = args.batch
load_pretrained = args.pretrained
load_from_checkpoint = args.precheck
model = args.model
project_name = args.project_name
weight_decay = args.decay
lr = args.lr
gamma = args.gamma
step = args.step
axis = args.axis

if load_pretrained and load_from_checkpoint:
    raise ValueError("You cannot set both --pretrained and --precheck at the same time.")
elif not load_pretrained and not load_from_checkpoint:
    print("Neither pretrained nor checkpoint selected — starting from scratch.")

predicting = Trainer(main_path,
            model_id=int(model),
            batch_size=int(batch_size),
            epochs=int(epochs),
            cuda_id=int(cuda_id),
            load_from_checkpoint=load_from_checkpoint,
            load_pretrained=load_pretrained,
            project_name=str(project_name),
            lr=lr,
            gamma=gamma,
            weight_decay=weight_decay,
            step=step,
            axis=axis)

print('Finished')