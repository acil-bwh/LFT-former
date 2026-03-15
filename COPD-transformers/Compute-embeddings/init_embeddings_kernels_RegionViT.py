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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.models.region_ViT import RegionViT

# Dataset class
class GroupedSliceDataset(Dataset):
    def __init__(self, file_list, transform, slices_per_patient=9, img_dir="", label_cols=None, add=0):
        """
        file_list: pandas dataframe with image file names and labels
        label_cols: list of columns in dataframe corresponding to tasks
        """
        self.groups = []
        self.transform = transform
        self.img_dir = img_dir
        self.slices_per_patient = slices_per_patient
        self.label_cols = label_cols
        self.add = add

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

            if img_path.endswith(".png"):
                img = Image.open(img_path).convert('RGB')  # 3-channels
                img = transforms.ToTensor()(img)         # 3 x H x W

            elif img_path.endswith(".npy"):
                arr = np.load(img_path).astype(np.float32)  # H x W
                arr = np.expand_dims(arr, axis=0)           # 1 x H x W
                img = torch.from_numpy(arr)
                img = img.repeat(3, 1, 1)

            else:
                raise ValueError("Unsupported file type: " + img_path)
        
            imgs.append(self.transform(img))

        imgs = torch.stack(imgs, dim=0)

        # Labels per task as tuple
        if self.label_cols is not None:
            labels = tuple(torch.tensor(row[col]).float() for col in self.label_cols)
        else:
            labels = (torch.tensor(int(group.iloc[0, 1]) + self.add),)

        return imgs, labels
    
# Read CSV files
def read_files(file_path, num_slices, model_id, project_name):
    diagnosis = ["File name","emph_cat_P1","traj","finalgold_visit_P1"]
    if model_id == 1 or model_id == 3:
        model_name = "COPDEmph"
    elif model_id == 2:
        model_name = "TRAJ"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")
    
    if num_slices == 9:
        test_list = pd.read_csv(file_path+f'/common_kernel_{model_name}_{project_name}.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
    elif num_slices == 20:
        test_list = pd.read_csv(file_path+f'/common_kernel_{model_name}_{project_name}_all.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)

    return test_list

from einops import reduce

# Feature extraction
def extract_features_pre_logits(model, x):
    with torch.no_grad():
        local_tokens = model.local_encoder(x)
        region_tokens = model.region_encoder(x)

        for down, peg, transformer in model.layers:
            local_tokens = down(local_tokens)
            region_tokens = down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)

        # Pool region tokens to a vector of size last_dim
        pooled = reduce(region_tokens, 'b c h w -> b c', 'mean')
        features_before_logits = model.to_logits[1](pooled)  # LayerNorm

    return features_before_logits

def extract_vit_ready_features_by_patient(model, dataloader, device, num_slices,
                                          n_augmentations=10, save_dir=None, set_name="1"):
    model.eval()
    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.ToTensor()])

    all_features, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting kernel {set_name} features"):
            B, N, C, H, W = imgs.shape
            assert B == 1, "Batch size must be 1"
            assert N == num_slices, f"Expected {num_slices} slices per patient, got {N}"

            imgs = imgs.view(B * N, C, H, W).to(device)

            # Apply augmentations
            aug_features = []
            for _ in range(n_augmentations):
                pil_imgs = [transforms.ToPILImage()(img.cpu()) for img in imgs]
                aug_imgs = torch.stack([augment_transform(img) for img in pil_imgs]).to(device)
                layer_features = extract_features_pre_logits(model, aug_imgs)
                aug_features.append(layer_features)

            aug_features = torch.stack(aug_features, dim=1)  # (B*N, n_augmentations, D)
            aug_features = aug_features.view(B, N, n_augmentations, -1)  # (B, N, n_augmentations, D)
            all_features.append(aug_features.cpu())

            # Combine labels into a single tensor (num_tasks,)
            combined_labels = torch.stack([torch.tensor(lbl) for lbl in labels], dim=0).view(-1)
            all_labels.append(combined_labels.cpu())

    X = torch.cat(all_features, dim=0).numpy()
    y = torch.stack(all_labels, dim=0).numpy()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"test_features_k{set_name}.npy"), X)
        np.save(os.path.join(save_dir, f"test_labels_k{set_name}.npy"), y)
        print(f"{set_name} features and labels saved to {save_dir}")

    return X, y

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Main project path")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--slices", type=int, default=9)
    parser.add_argument("--model", type=int, required=True)
    parser.add_argument("--project_name", type=str, default="COPDGene")
    parser.add_argument("--n_augmentations", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.path, f"{args.project_name}{args.model}-features_kernels-{args.slices}")
    for kernel in [1,2,3,4,5,6]:
        img_dir = os.path.join(args.path, f"{args.project_name}k{kernel}-imgs")
        files_dir = os.path.join(args.path, f"{args.project_name}-files")
        model_path = os.path.join(args.path, f"{args.project_name}-checkpoints/checkpoints-RegionViT/", f"best_{args.project_name}_{args.model}.pt")
        os.makedirs(save_dir, exist_ok=True)

        if args.model == 1: num_classes=4
        else: num_classes=6
        
        dim = (128, 256, 512, 1024)
        depth = (2, 2, 14, 2)
        patch_size = 4
        window_size = 7
        dropout = 0.1
        emb_dropout = 0.1
        channels = 3
        RegViT_model = RegionViT(dim=dim, depth=depth, local_patch_size=patch_size,
                                window_size=window_size, num_classes=num_classes)

        print(f"Loading from path: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        RegViT_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        RegViT_model.to(device)
        RegViT_model.eval()

        if args.model == 1: add=0
        elif args.model == 2: add=-1
        elif args.model == 3: add=1

        label_cols = ["emph_cat_P1","traj","finalgold_visit_P1"]
        apply_transforms = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.RandomRotation(20)])
        test_list = read_files(files_dir, args.slices, args.model, args.project_name)
        test_data = GroupedSliceDataset(test_list, transform=apply_transforms, slices_per_patient=args.slices, img_dir=img_dir, label_cols=label_cols, add=add)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        X_valid, y_valid = extract_vit_ready_features_by_patient(
            model=RegViT_model, dataloader=test_loader, device=device, num_slices=args.slices,
            n_augmentations=args.n_augmentations, save_dir=save_dir, set_name=kernel)

        print(f"Feature extraction completed for kernel {kernel}:")