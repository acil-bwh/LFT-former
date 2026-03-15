# ············· QUERALT MARTÍN-SALADICH, 2024 ··············
# ···················· BWH + UPF + VHIR ····················
# ··························································

from Utils.training.training_model_augViT import Trainer_augViT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path", required=True)

parser.add_argument("--cuda", type=int, help="cuda node id", default=0)
parser.add_argument("--epochs", type=int, help="epochs",default=500)
parser.add_argument("--batch", type=int, help="batch size: 64 OR 32",default=32)
parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights')
parser.add_argument('--precheck', action='store_true', help='Load from checkpoint')
parser.add_argument("--model", type=int, help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)
parser.add_argument("--project_name", type=str, help="project dataset name, e.g. COPD",default="COPDGene")
parser.add_argument("--slices", type=int, help="9 or 20 slices",default=20)
parser.add_argument("--lr", type=float, help="learning rate",default=1e-5)
parser.add_argument("--decay", type=float, help="weight decay",default=1e-4)
parser.add_argument("--gamma", type=float, help="gamma",default=0.9)
parser.add_argument("--step", type=int, help="step",default=5)

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
epochs = args.epochs
batch_size = args.batch
load_pretrained = args.pretrained
load_from_checkpoint = args.precheck
model = args.model
project_name = args.project_name
slices = args.slices
weight_decay = args.decay
lr = args.lr
gamma = args.gamma
step = args.step

if load_pretrained and load_from_checkpoint:
    raise ValueError("You cannot set both --pretrained and --precheck at the same time.")
elif not load_pretrained and not load_from_checkpoint:
    print("Neither pretrained nor checkpoint selected — starting from scratch.")

predicting = Trainer_augViT(main_path,
            model_id=model,
            num_slices=slices,
            cuda_id=cuda_id,
            project_name=project_name,
            epochs=epochs,
            batch_size=batch_size,
            load_pretrained=load_pretrained,
            load_from_checkpoint=load_from_checkpoint,
            lr=lr,
            gamma=gamma,
            weight_decay=weight_decay,
            step=step)

print('Finished')