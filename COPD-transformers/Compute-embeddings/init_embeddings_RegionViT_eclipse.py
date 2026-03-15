# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from einops import reduce
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.models.region_ViT import RegionViT  # Adjust this path if needed
from scipy.ndimage import median_filter

# ---------------- Dataset ----------------
class GroupedSliceDataset(Dataset):
    def __init__(self, file_list, transform, slices_per_patient=20, img_dir="", label_cols=None, add=0, filter_size=3):
        self.groups = []
        self.transform = transform
        self.img_dir = img_dir
        self.slices_per_patient = slices_per_patient
        self.label_cols = label_cols
        self.add = add
        self.filter_size = filter_size

        # Group rows by patient (20 slices each)
        for i in range(0, len(file_list), slices_per_patient):
            group = file_list.iloc[i:i + slices_per_patient]
            if len(group) == slices_per_patient:
                self.groups.append(group)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        imgs = []
        for _, row in group.iterrows():
            img_path = os.path.join(self.img_dir, row["File name"])

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing file: {img_path}")

            # Optimized .npy loading for RegionViT (expects 3 channels)
            arr = np.load(img_path).astype(np.float32)
            if self.filter_size != 0:
                arr = median_filter(arr, size=self.filter_size)

            img = torch.from_numpy(arr).unsqueeze(0) # 1 x H x W
            img = img.repeat(3, 1, 1)                # 3 x H x W
        
            imgs.append(self.transform(img))

        imgs = torch.stack(imgs, dim=0)

        # Apply label adjustment 'add' to the traj column
        if self.label_cols is not None:
            # Assumes traj is the label to be adjusted
            label_v1 = torch.tensor(group.iloc[0]["traj"] + self.add).float()
            label_v2 = torch.tensor(group.iloc[0]["traj_v2"] + self.add).float()
        else:
            label_v1 = torch.tensor(int(group.iloc[0, 1]) + self.add).float()
            label_v2 = torch.tensor(int(group.iloc[0, 2]) + self.add).float()

        return imgs, label_v1, label_v2

# ---------------- CSV Loader ----------------
def read_files(file_path, num_slices):
    diagnosis = ["File name", "traj", "traj_v2"]
    # Only supporting 20 slices as per your requirement
    if num_slices == 20:
        csv_name = f'testing_TRAJ_ECLIPSE_all.csv'
        full_path = os.path.join(file_path, csv_name)
        test_list = pd.read_csv(full_path, sep=",", usecols=diagnosis, header=0, low_memory=False)
    else:
        raise ValueError("This script is configured specifically for 20 slices.")

    return test_list

# ---------------- Feature extraction ----------------
def extract_features_pre_logits(model, x):
    # x shape: (N, 3, 224, 224)
    local_tokens = model.local_encoder(x)
    region_tokens = model.region_encoder(x)

    for down, peg, transformer in model.layers:
        local_tokens = down(local_tokens)
        region_tokens = down(region_tokens)
        local_tokens = peg(local_tokens)
        local_tokens, region_tokens = transformer(local_tokens, region_tokens)

    pooled = reduce(region_tokens, 'b c h w -> b c', 'mean')
    # Accessing LayerNorm before the final linear head
    features_before_logits = model.to_logits[1](pooled) 

    return features_before_logits

def extract_vit_ready_features_by_patient(model, dataloader, device, num_slices,
                                          n_augmentations=10, save_dir=None, split_name="test", version=""):
    model.eval()
    # Augmentation transform for Tensors
    aug_transform = transforms.RandomRotation(20)

    all_features, all_labels, all_labels_v2 = [], [], []

    with torch.no_grad():
        for imgs, label_v1, label_v2 in tqdm(dataloader, desc=f"Extracting {split_name} features"):
            # imgs: (1, 20, 3, 224, 224)
            B, N, C, H, W = imgs.shape
            imgs = imgs.view(N, C, H, W).to(device)

            aug_features_list = []
            for _ in range(n_augmentations):
                # Faster: apply rotation directly to tensor
                aug_imgs = aug_transform(imgs)
                feat = extract_features_pre_logits(model, aug_imgs) # (N, D)
                aug_features_list.append(feat.cpu())

            # Stack to (N, n_aug, D) then add Batch dim -> (1, N, n_aug, D)
            aug_features = torch.stack(aug_features_list, dim=1).unsqueeze(0)
            all_features.append(aug_features)
            all_labels.append(label_v1)
            all_labels_v2.append(label_v2)

    X = torch.cat(all_features, dim=0).numpy()
    y_v1 = torch.stack(all_labels, dim=0).numpy()
    y_v2 = torch.stack(all_labels_v2, dim=0).numpy()

    if save_dir:
        np.save(os.path.join(save_dir, f"{split_name}_features{version}.npy"), X)
        np.save(os.path.join(save_dir, f"{split_name}_labels_v1.npy"), y_v1)
        np.save(os.path.join(save_dir, f"{split_name}_labels_v2.npy"), y_v2)

    return X, y_v1, y_v2

# ---------------- Main script ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Main project path")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--model_id", type=int, default=2)
    parser.add_argument("--slices", type=int, default=20)
    parser.add_argument("--project_name", type=str, default="ECLIPSE")
    parser.add_argument("--n_augmentations", type=int, default=10)
    parser.add_argument("--axis", type=str, default="axial")
    parser.add_argument("--v", type=str, default="", help=" "" for filtered, 0 for unfiltered")
    args = parser.parse_args()

    if args.v == "0":
        filter_size = 0
    else:
        filter_size = 3

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    files_dir = os.path.join(args.path, f"COPDGene-files")

    if args.axis == "coronal":
        img_dir = os.path.join(args.path, f"{args.project_name}-imgs-cor")
        model_path = os.path.join(args.path, f"COPDGene-checkpoints/checkpoints-RegionViT-cor/", f"best_RegionViT_{args.model_id}_cor.pt")
        save_dir = os.path.join(args.path, f"ECLIPSE{args.model_id}-features-{args.slices}-cor")
    elif args.axis == "axial":
        img_dir = os.path.join(args.path, f"{args.project_name}-imgs")
        model_path = os.path.join(args.path, f"COPDGene-checkpoints/checkpoints-RegionViT/", f"best_RegionViT_{args.model_id}.pt")
        save_dir = os.path.join(args.path, f"ECLIPSE{args.model_id}-features-{args.slices}")

    os.makedirs(save_dir, exist_ok=True)

    # Define RegionViT model parameters
    if args.model_id == 1:
        num_classes = 4
    else:
        num_classes = 6
        
    dim = (128, 256, 512, 1024)
    depth = (2, 2, 14, 2)
    patch_size = 4
    window_size = 7
    RegViT_model = RegionViT(dim=dim, depth=depth, local_patch_size=patch_size,
                             window_size=window_size, num_classes=num_classes)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    RegViT_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    RegViT_model.to(device)
    RegViT_model.eval()

    add_dict = {1:0, 2:-1, 3:1}
    current_add = add_dict.get(args.model_id, 0)

    base_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True)])

    test_list = read_files(files_dir, args.slices)
    
    test_data = GroupedSliceDataset(
        test_list, 
        transform=base_transform, 
        slices_per_patient=args.slices, 
        img_dir=img_dir, 
        label_cols=["traj","traj_v2"], 
        add=current_add,
        filter_size=filter_size)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    X_test, y_test_v1, y_test_v2 = extract_vit_ready_features_by_patient(
        RegViT_model, test_loader, device, num_slices=args.slices,
        n_augmentations=args.n_augmentations, save_dir=save_dir, split_name="test", version=args.v)

print(" FINISHED ")