# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
#       ···················· BWH  ····················
# ··························································

from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------- SEED SETUP ----------------
def seed_everything(seed: int = 3):
    """Ensure full reproducibility across runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(3)

# ---------------- PREDICTOR FUNCTION ----------------
def Predictor_COPD(main_path,
                   model_id,
                   num_slices,
                   cuda_id,
                   output_id,
                   project_name="COPD"):
    """
    Runs multi-slice Vision Transformer predictions for all patients.
    Handles per-patient aggregation automatically.

    Args:
        main_path (str): Root directory.
        model_id (int): Model type (1=Emphysema, 2=Trajectories, 3=COPD).
        num_slices (int): Number of slices per patient (e.g. 9 or 20).
        cuda_id (int): GPU ID to use.
        output_id (int): Model type (1=Emphysema, 2=Trajectories, 3=COPD).
        project_name (str): Project name prefix for folder structure.
    """

    # ---------------- PATHS ----------------
    cuda_set = f"cuda:{cuda_id}"
    save_path = os.path.join(main_path, f"{project_name}-results/models")
    models_path = os.path.join(main_path, f"{project_name}-models")

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(cuda_set if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # ---------------- MODEL CONFIG ----------------
    if model_id == 1:
        print(f"···· EMPHYSEMA MODEL {num_slices}-slices ····")
        num_classes = 4
        add = 0
    elif model_id == 2:
        print(f"···· TRAJECTORIES MODEL {num_slices}-slices ····")
        num_classes = 6
        add = -1
    elif model_id == 3:
        print(f"···· COPD MODEL {num_slices}-slices ····")
        num_classes = 6
        add = 1
    else:
        raise ValueError("Invalid model_id (must be 2, or 3).")

    if num_slices == 9:
        fusion_type = "C"
        added = ""
    elif num_slices == 20:
        fusion_type = "W"
        added = "all"
    else:
        raise ValueError("Invalid num_slices (must be 9 or 20).")

    # ---------------- DATASET ----------------
    class FeatureDataset(Dataset):
        def __init__(self, features_path, labels_path):
            self.features = np.load(features_path)  # (B, num_slices, n_augmentations, last_dim)
            self.labels = np.load(labels_path)      # (B,)

            if self.features.ndim != 4:
                raise ValueError(f"Expected features shape (B, N, A, D) but got {self.features.shape}")
            if self.labels.ndim != 1:
                raise ValueError(f"Labels should be 1D but got {self.labels.shape}")

            self.num_patients = self.features.shape[0]
            self.num_slices = self.features.shape[1]
            self.num_augmentations = self.features.shape[2]

        def __len__(self):
            return self.num_patients

        def __getitem__(self, idx):
            aug_idx = np.random.randint(self.num_augmentations)
            feature = self.features[idx, :, aug_idx, :]  # (num_slices, last_dim)
            
            feature = np.expand_dims(feature, axis=1)

            # Convert to torch tensor
            feature = torch.from_numpy(feature).float()
            label = torch.tensor(self.labels[idx]).long()

            return feature, label
        
    source_features = f"{project_name}{model_id}-features-{num_slices}"
    features_dir = os.path.join(main_path, source_features)

    valid_features_path = os.path.join(features_dir, "test_features.npy")

    source_labels = f"{project_name}{output_id}-features-{num_slices}"
    labels_dir = os.path.join(main_path, source_labels)
    valid_labels_path = os.path.join(labels_dir,f"test_labels_{output_id}.npy")

    valid_dataset = FeatureDataset(valid_features_path, valid_labels_path)
    test_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # ---------------- MODEL LOADING ----------------
    from Utils.models.basicViT import ViT

    dim = (128, 256, 512, 1024)
    vec_dim = dim[-1]

    model = ViT(
        num_vectors=num_slices,
        vec_dim=vec_dim,
        num_classes=num_classes,
        dim=512,
        depth=4,
        heads=4,
        mlp_dim=1024,
        pool="cls").to(device)

    models_path = f"{main_path}/{project_name}-checkpoints/checkpoints-augViT"
    checkpoint_path = os.path.join(models_path, f"best_augViT_{project_name}_{model_id}{fusion_type}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---------------- STATS PREP ----------------
    added = [0,-1,1]
    test_labels = np.array(valid_dataset.labels).astype(int) + added[output_id-1]
    test_labels[test_labels==-1] = 0
    num_classes = len(np.unique(test_labels))
    n_samples = len(test_labels)

    print(f"Number of labels = {num_classes}")
    print(f"Number of samples = {n_samples}")
    print(f"Labels = {np.unique(test_labels)}")

    counts = np.bincount(test_labels)
    proportion = counts/n_samples
    print(f"Proportions: {proportion}")
    proportion = torch.tensor(1.0/proportion, dtype=torch.float32, device=device)

    real = np.zeros(n_samples)
    probable = np.zeros(n_samples)

    # # ---------------- INFERENCE ----------------
    with torch.no_grad():
        for ix, (features, label) in enumerate(tqdm(test_loader, desc="Testing patients")):
            features = features.to(device)
            label = label.to(device) + added[output_id-1]
            label[label == -1] = 0

            logits = model(features.squeeze(2))  # [B, C]
            probs = torch.softmax(logits, dim=-1)  # [B, C]
            
            probs = probs*1.0/proportion
            raw_pred = torch.argmax(probs, dim=1)

            # 4. Storage
            # Ensure probabs array is initialized for 6 columns
            real[ix] = label.item()
            probable[ix] = raw_pred.item()

    # ---------------- SAVE RESULTS ----------------
    base_csv = os.path.join(save_path, f"model_augViT_{project_name}_i{model_id}-o{output_id}_original.csv")

    # prediction summary per patient
    df = pd.DataFrame({
        "data": np.arange(n_samples),
        "label": real.astype(int),
        "predicted": probable.astype(int)})
    df.to_csv(base_csv, index=False)

    print(f"\nFINISHED PREDICTING VALUES AND PROBABILITIES for {n_samples} patients - 6 CLASSES.")
    print(f"Saved results:\n  {base_csv}\n")

    # ################################################

    real = np.zeros(n_samples)
    probable = np.zeros(n_samples)

    # # ---------------- INFERENCE ----------------
    with torch.no_grad():
        for ix, (features, label) in enumerate(tqdm(test_loader, desc="Testing patients")):
            features = features.to(device)
            label = label.to(device) + added[output_id-1]
            label[label == -1] = 1
            label[label == 0] = 1

            logits = model(features.squeeze(2))  # [B, C]
            probs = torch.softmax(logits, dim=-1)  # [B, C]
            
            probs = probs*1.0/proportion
            raw_pred = torch.argmax(probs, dim=1)

            if raw_pred.item() == 0 or raw_pred.item() == 1:
                final_pred = 1
            else:
                final_pred = raw_pred.item()

            # 4. Storage
            # Ensure probabs array is initialized for 5 columns, not 6
            real[ix] = label.item()
            probable[ix] = final_pred

    # ---------------- SAVE RESULTS ----------------
    base_csv = os.path.join(save_path, f"model_augViT_{project_name}_i{model_id}-o{output_id}_5class.csv")

    # prediction summary per patient
    df = pd.DataFrame({
        "data": np.arange(n_samples),
        "label": real.astype(int),
        "predicted": probable.astype(int)})
    df.to_csv(base_csv, index=False)

    print(f"\nFINISHED PREDICTING VALUES AND PROBABILITIES for {n_samples} patients - 5 CLASSES.")
    print(f"Saved results:\n  {base_csv}\n")

    # ################################################

    real = np.zeros(n_samples)
    probable = np.zeros(n_samples)

    # ---------------- INFERENCE ----------------
    with torch.no_grad():
        for ix, (features, label) in enumerate(tqdm(test_loader, desc="Testing patients")):
            features = features.to(device)
            label = label.to(device) + added[output_id-1]
            binary_label = 0 if label.item() <= 1 else 1        

            logits = model(features.squeeze(2))  # [B, C]
            probs = torch.softmax(logits, dim=-1)  # [B, C]

            probs = probs*1.0/proportion

            # 3. Predicted class based on 5-class space
            raw_pred = torch.argmax(probs, dim=1)

            if raw_pred.item() == 0 or raw_pred.item() == 1:
                final_pred = 0
            else:
                final_pred = 1
            
            real[ix] = binary_label
            probable[ix] = final_pred

    # ---------------- SAVE RESULTS ----------------
    base_csv = os.path.join(save_path, f"model_augViT_{project_name}_i{model_id}-o{output_id}_binary.csv")

    # prediction summary per patient
    df = pd.DataFrame({
        "data": np.arange(n_samples),
        "label": real.astype(int),
        "predicted": probable.astype(int)})
    df.to_csv(base_csv, index=False)

    print(f"\nFINISHED PREDICTING VALUES AND PROBABILITIES for {n_samples} patients - 2 CLASSES.")
    print(f"Saved results:\n  {base_csv}\n")

    return
