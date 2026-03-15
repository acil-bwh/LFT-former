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

# ---------------- Dataset ----------------
class GroupedSliceDataset(Dataset):
    def __init__(self, file_list, transform, slices_per_patient=20, img_dir="", label_cols=None, add=0):
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

# ---------------- CSV Loader ----------------
def read_files(file_path, num_slices, model_id, project_name, axis):
    diagnosis = ["File name","emph_cat_P1","traj","finalgold_visit_P1","PRM_pct_airtrapping_Thirona_P1"]
    if model_id in [1,3,4,5,6]:
        model_name = "COPDEmph"
    elif model_id == 2:
        model_name = "TRAJ"
    else:
        raise ValueError(f"Unsupported model id: {model_id}")

    if num_slices == 9:
        if axis == "coronal":
            raise ValueError(f"Can only be the original axial model")
        elif axis == "axial":
            train_list = pd.read_csv(file_path+f'/training_{model_name}_{project_name}.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
            val_list = pd.read_csv(file_path+f'/validation_{model_name}_{project_name}.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
            test_list = pd.read_csv(file_path+f'/testing_{model_name}_{project_name}.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
    
    elif num_slices == 20:
        if axis == "coronal":
            train_list = pd.read_csv(file_path+f'/training_{model_name}_{project_name}_cor.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
            val_list = pd.read_csv(file_path+f'/validation_{model_name}_{project_name}_cor.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
            test_list = pd.read_csv(file_path+f'/testing_{model_name}_{project_name}_cor.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
        elif axis == "axial":
            train_list = pd.read_csv(file_path+f'/training_{model_name}_{project_name}_all.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
            val_list = pd.read_csv(file_path+f'/validation_{model_name}_{project_name}_all.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
            test_list = pd.read_csv(file_path+f'/testing_{model_name}_{project_name}_all.csv', sep=",", usecols=diagnosis,header=0,low_memory=False)
    
    if model_id == 3:
        target_col = "finalgold_visit_P1"
        def bin_labels(val):
            if val < -1:
                return 0
            else: 
                return val

        train_list[target_col] = train_list[target_col].apply(bin_labels)
        val_list[target_col] = val_list[target_col].apply(bin_labels)
        test_list[target_col] = test_list[target_col].apply(bin_labels)

    if model_id == 4:
        target_col = "PRM_pct_airtrapping_Thirona_P1"
        def bin_labels(val):
            if val <= 10:
                return 0
            elif 10 < val <= 20:
                return 1
            elif 20 < val <= 40:
                return 2
            else: # > 40
                return 3

        train_list[target_col] = train_list[target_col].apply(bin_labels)
        val_list[target_col] = val_list[target_col].apply(bin_labels)
        test_list[target_col] = test_list[target_col].apply(bin_labels)

    if model_id == 5:
        target_col = "finalgold_visit_P1"
        def bin_labels(val):
            if val <= 0:
                return 0
            else: 
                return val

        train_list[target_col] = train_list[target_col].apply(bin_labels)
        val_list[target_col] = val_list[target_col].apply(bin_labels)
        test_list[target_col] = test_list[target_col].apply(bin_labels)

    if model_id == 6:
        target_col = "finalgold_visit_P1"
        def bin_labels(val):
            if val <= 0:
                return 0
            else: 
                return 1

        train_list[target_col] = train_list[target_col].apply(bin_labels)
        val_list[target_col] = val_list[target_col].apply(bin_labels)
        test_list[target_col] = test_list[target_col].apply(bin_labels)

    return train_list, val_list, test_list

# ---------------- Feature extraction ----------------
def extract_features_pre_logits(model, x):
    with torch.no_grad():
        local_tokens = model.local_encoder(x)
        region_tokens = model.region_encoder(x)

        for down, peg, transformer in model.layers:
            local_tokens = down(local_tokens)
            region_tokens = down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)

        pooled = reduce(region_tokens, 'b c h w -> b c', 'mean')
        features_before_logits = model.to_logits[1](pooled)  # LayerNorm

    return features_before_logits

# ---------------- Main feature extraction function ----------------
def extract_vit_ready_features_by_patient(model, dataloader, device,
                                          n_augmentations=None, save_dir=None, split_name="train"):
    """
    Iterates over patients to extract embeddings.
    If n_augmentations is specified, it performs Test-Time Augmentation (TTA).
    """
    model.eval()
    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.ToTensor()])

    all_features, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting {split_name} features"):
            B, N, C, H, W = imgs.shape
            assert B == 1, "Batch size must be 1 to maintain patient-slice grouping"

            imgs = imgs.view(B * N, C, H, W).to(device)

            # Scenario A: Test-Time Augmentation (TTA)
            if n_augmentations is not None:
                aug_features = []
                for _ in range(n_augmentations):
                    pil_imgs = [transforms.ToPILImage()(img.cpu()) for img in imgs]
                    aug_imgs = torch.stack([augment_transform(img) for img in pil_imgs]).to(device)
                    layer_features = extract_features_pre_logits(model, aug_imgs)
                    aug_features.append(layer_features)

                # Reshape to [B, Slices, Augmentations, Hidden_Dim]
                aug_features = torch.stack(aug_features, dim=1)
                aug_features = aug_features.view(B, N, n_augmentations, -1)
                all_features.append(aug_features.cpu())

            # Scenario B: Standard single-pass extraction
            else:
                feat = extract_features_pre_logits(model, imgs)
                feat = feat.view(B, N, 1, -1)

                all_features.append(feat.cpu())

            # Consolidate labels into a single patient-level tensor
            combined_labels = torch.stack([torch.tensor(lbl) for lbl in labels], dim=0).view(-1)
            all_labels.append(combined_labels.cpu())

    X = torch.cat(all_features, dim=0).numpy()    
    y = torch.stack(all_labels, dim=0).numpy()

    # Persistence: Save to NPY for training script ingestion
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{split_name}_features.npy"), X)
        np.save(os.path.join(save_dir, f"{split_name}_labels.npy"), y)

    return X, y

# ---------------- Main script ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Main project path")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--model_id", type=int, default=2)
    parser.add_argument("--slices", type=int, default=9)
    parser.add_argument("--project_name", type=str, default="COPDGene")
    parser.add_argument("--n_augmentations", type=int, default=None)
    parser.add_argument("--axis", type=str, default="axial")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    files_dir = os.path.join(args.path, f"{args.project_name}-files")

    # Define RegionViT model parameters
    if args.model_id == 1:
        num_classes = 4
        window_type = "emph"
    if args.model_id == 2:
        num_classes = 6
        window_type = "lung"
    elif args.model_id == 3:
        num_classes = 6
        window_type = "copd"
    elif args.model_id == 4:
        num_classes = 4
        window_type = "trap"
    elif args.model_id == 5:
        num_classes = 5
        window_type = "copd"
    elif args.model_id == 6:
        num_classes = 2
        window_type = "copd"

    if args.axis == "coronal":
        img_dir = os.path.join(args.path, f"{args.project_name}-{window_type}-cor")
        model_path = os.path.join(args.path, f"{args.project_name}-checkpoints/checkpoints-RegionViT-cor/", f"best_RegionViT_{args.model_id}_cor.pt")
        save_dir = os.path.join(args.path, f"COPDGene-embeddings/{args.project_name}{args.model_id}-features-{args.slices}-cor")
    elif args.axis == "axial":
        img_dir = os.path.join(args.path, f"{args.project_name}-{window_type}")
        model_path = os.path.join(args.path, f"{args.project_name}-checkpoints/checkpoints-RegionViT/", f"best_RegionViT_{args.model_id}.pt")
        save_dir = os.path.join(args.path, f"COPDGene-embeddings/{args.project_name}{args.model_id}-features-{args.slices}")

    os.makedirs(save_dir, exist_ok=True)
        
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

    # Label adjustment based on model
    add = {1:0, 2:-1, 3:1, 4:0, 5:0, 6:0}

    # Read CSVs
    train_list, val_list, test_list = read_files(files_dir, args.slices, args.model_id, args.project_name, args.axis)

    apply_transforms = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.RandomRotation(20)])
    
    label_cols = ["emph_cat_P1","traj","finalgold_visit_P1","PRM_pct_airtrapping_Thirona_P1"]

    train_data = GroupedSliceDataset(train_list, transform=apply_transforms, slices_per_patient=args.slices, img_dir=img_dir, label_cols=label_cols, add=add)
    val_data = GroupedSliceDataset(val_list, transform=apply_transforms, slices_per_patient=args.slices, img_dir=img_dir, label_cols=label_cols, add=add)
    test_data = GroupedSliceDataset(test_list, transform=apply_transforms, slices_per_patient=args.slices, img_dir=img_dir, label_cols=label_cols, add=add)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Extract features
    X_train, y_train = extract_vit_ready_features_by_patient(
        RegViT_model, train_loader, device,
        n_augmentations=args.n_augmentations, save_dir=save_dir, split_name="train")

    X_val, y_val = extract_vit_ready_features_by_patient(
        RegViT_model, val_loader, device,
        n_augmentations=args.n_augmentations, save_dir=save_dir, split_name="valid")
    
    X_test, y_test = extract_vit_ready_features_by_patient(
        RegViT_model, test_loader, device,
        n_augmentations=args.n_augmentations, save_dir=save_dir, split_name="test")

    print("Feature extraction completed:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)