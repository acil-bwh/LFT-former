# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- Reproducibility ----------------------
def seed_everything(seed=3):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(3)

# ---------------------- Prediction Function ----------------------
def Probabilities_augViT(
        main_path,
        output_id,
        num_slices,
        cuda_id,
        patient_sid,
        project_name):
    
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    if output_id == 1:
        print(f"···· EMPHYSEMA MODEL {num_slices}-slices ····")
        num_classes = 4
        add = 0
        model_id = 3
    elif output_id == 2:
        print(f"···· TRAJECTORIES MODEL {num_slices}-slices ····")
        num_classes = 6
        add = -1
        model_id = 3
    elif output_id == 3:
        print(f"···· COPD MODEL {num_slices}-slices ····")
        num_classes = 6
        add = 1
        model_id = 3
    elif output_id == 4:
        print(f"···· TRAPPING MODEL {num_slices}-slices ····")
        num_classes = 4
        add = 1
        model_id = 3
    elif output_id == 5:
        print(f"···· COPD-5 MODEL {num_slices}-slices ····")
        num_classes = 5
        add = 0
        model_id = 3
    elif output_id == 6:
        print(f"···· COPD-2 MODEL {num_slices}-slices ····")
        num_classes = 2
        add = 0
        model_id = 3
    else:
        raise ValueError("Invalid model_id (must be 1...6).")

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
        def __init__(self, features_path, labels_path, csv_path, model_id):
            self.features = np.load(features_path)  # (B, num_slices, n_augmentations, last_dim)
            self.labels = np.load(labels_path)      # (B,)
            self.model_id = model_id
            self.labels = self.labels[:,self.model_id-1]
            df = pd.read_csv(csv_path)
            if "sid" not in df.columns:
                df.rename(columns={df.columns[0]: "sid"}, inplace=True)

            if self.features.ndim != 4:
                raise ValueError(f"Expected features shape (B, N, A, D) but got {self.features.shape}")
            if self.labels.ndim != 1:
                raise ValueError(f"Labels should be 1D but got {self.labels.shape}")

            self.sids = df["sid"].astype(str).tolist()
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
        
    source_features = f"{project_name}-embeddings/{project_name}{output_id}-features-{num_slices}"
    features_dir = os.path.join(main_path, source_features)

    valid_features_path = os.path.join(features_dir, "test_features.npy")

    source_labels = f"{project_name}-embeddings/{project_name}{output_id}-features-{num_slices}"
    labels_dir = os.path.join(main_path, source_labels)
    
    csv_path = f"{main_path}{project_name}-files/df_val_copd.csv"
    valid_labels_path = os.path.join(labels_dir,f"test_labels.npy")

    valid_dataset = FeatureDataset(valid_features_path, valid_labels_path, csv_path, model_id)
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
        depth=8,
        heads=4,
        mlp_dim=1024,
        pool="cls").to(device)

    models_path = f"{main_path}/{project_name}-checkpoints/checkpoints-augViT"
    checkpoint_path = os.path.join(models_path, f"best_augViT_{project_name}_{output_id}{fusion_type}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---------------- STATS PREP ----------------
    test_labels = np.array(valid_dataset.labels).astype(int)
    test_labels[test_labels==-2] = 0

    if output_id == 5:
        test_labels[test_labels == -1] = 0

    elif output_id == 6:
        test_labels[test_labels == -1] = 0
        test_labels[test_labels > 0] = 1

    test_labels = test_labels + add

    num_classes = len(np.unique(test_labels))
    n_samples = len(test_labels)

    print(f"Number of labels = {num_classes}")
    print(f"Number of samples = {n_samples}")
    print(f"Labels = {np.unique(test_labels)}")

    counts = np.bincount(test_labels)
    proportion = counts/n_samples
    print(f"Proportions: {proportion}")

    try:
        idx = valid_dataset.sids.index(str(patient_sid))
    except ValueError:
        print(f"ID {patient_sid} not found."); return

    data = valid_dataset[idx]
    with torch.no_grad():
        feat = data[0].unsqueeze(0).squeeze(2).to(device)
        label = data[1]
        label = label.to(device)
        label[label==-2]=0

        if output_id == 5:
            label[label == -1] = 0

        elif output_id == 6:
            label[label == -1] = 0
            label[label > 0] = 1
        
        label = label + add
        logits = model(feat)

        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        # --- Predicted class ---
        pred = np.argmax(probs) 
        true_idx = int(label)

    print(f"\n{'='*50}")
    print(f"RESULTS FOR PATIENT: {id}")
    print(f"{'='*50}")
    print(f"{'Label':<12} | {'Raw Prob':<10}")
    print(f"{'-'*45}")
    
    for i in range(num_classes):
        # Calculate real-world category name
        cat_name = f"{output_id}_{i - add}"
        
        # Add Markers
        markers = []
        if i == pred: markers.append("[PREDICTED]")
        if i == true_idx: markers.append("[TRUE LABEL]")
        marker_str = " ".join(markers)
        
        print(f"{cat_name:<12} | {probs[i]:.4f} {marker_str}")

    print(f"{'-'*45}")
    print(f"Final Prediction: {output_id}_{pred - add}")
    print(f"Actual Value:     {output_id}_{true_idx - add}")
    print(f"Result:           {'CORRECT' if pred == true_idx else 'INCORRECT'}")
    print(f"{'='*50}\n")

    return probs