# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

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

def get_patient_index_from_id(file_list_df, patient_id):
    """
    Map a patient string ID to its integer index in the dataset.
    
    Args:
        file_list_df: pd.DataFrame containing slice filenames in the first column.
        patient_id: string, e.g., "COPD123"

    Returns:
        idx: integer index of the patient in the dataset
    """
    # Extract patient IDs from filenames
    patient_ids = (
        file_list_df.iloc[:, 0]
        .astype(str)
        .str.replace(".png", "", regex=False)
        .str.replace(r"\d+$", "", regex=True)  # remove trailing slice numbers
        .drop_duplicates()
        .tolist()
    )

    if patient_id not in patient_ids:
        raise ValueError(f"Patient ID '{patient_id}' not found in file list.")

    idx = patient_ids.index(patient_id)
    return idx

class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path, file_list_df, dict_out):
        """
        features_path: path to .npy file with shape (B, num_slices, n_augmentations, D)
        labels_path: path to .npy file with shape (B, num_tasks)
        file_list_df: pd.DataFrame with image filenames in first column (one per slice)
        dict_out: dict of task_name -> list of class names (for classification) or 1 (for regression)
        """
        self.features = np.load(features_path)  # (B, num_slices, n_augmentations, D)
        self.labels = np.load(labels_path)      # (B, num_tasks)

        # --- Validate ---
        if self.features.ndim != 4:
            raise ValueError(f"Expected features shape (B, N, A, D) but got {self.features.shape}")
        if self.labels.ndim != 2:
            raise ValueError(f"Labels should be 2D but got {self.labels.shape}")

        # --- Extract patient IDs from filenames ---
        self.patient_ids = (
            file_list_df.iloc[:, 0]
            .astype(str)
            .str.replace(".png", "", regex=False)
            .str.replace(r"\d+$", "", regex=True)
            .drop_duplicates()
            .tolist()
        )

        if len(self.patient_ids) != self.features.shape[0]:
            raise ValueError(
                f"Number of patient IDs ({len(self.patient_ids)}) must match number of feature entries ({self.features.shape[0]})."
            )

        # --- Metadata ---
        self.num_patients = self.features.shape[0]
        self.num_slices = self.features.shape[1]
        self.num_augmentations = self.features.shape[2]
        self.num_classes = tuple(len(v) if isinstance(v, list) else 1 for v in dict_out.values())

        # --- Mapping patient IDs to indices ---
        self.id_to_index = {pid: idx for idx, pid in enumerate(self.patient_ids)}

    def __len__(self):
        return self.num_patients

    def __getitem__(self, idx):
        # --- Select one augmentation per slice randomly ---
        aug_idx = np.random.randint(self.num_augmentations, size=self.num_slices)
        feature = self.features[idx, np.arange(self.num_slices), aug_idx, :]  # (num_slices, D)
        feature = np.expand_dims(feature, axis=1)
        feature = torch.from_numpy(feature).float()

        # --- Prepare labels as tuple per task ---
        labels_row = self.labels[idx]  # (num_tasks,)
        labels_tuple = tuple(
            torch.tensor(labels_row[i]).float() if self.num_classes[i] == 1 else torch.tensor(labels_row[i]).long()
            for i in range(len(labels_row))
        )

        pid = self.patient_ids[idx]
        return pid, feature, labels_tuple

    def get_patient_data(self, patient_id):
        """Return (patient_id, feature_tensor, label_tensor) for a given patient ID."""
        if patient_id not in self.id_to_index:
            raise ValueError(f"Patient ID '{patient_id}' not found in dataset")
        idx = self.id_to_index[patient_id]

        aug_idx = np.random.randint(self.num_augmentations, size=self.num_slices)
        feature = self.features[idx, np.arange(self.num_slices), aug_idx, :]
        feature = np.expand_dims(feature, axis=1)
        feature = torch.from_numpy(feature).float()

        labels_row = self.labels[idx]
        labels_tuple = tuple(
            torch.tensor(labels_row[i]).float() if self.num_classes[i] == 1 else torch.tensor(labels_row[i]).long()
            for i in range(len(labels_row))
        )

        return patient_id, feature, labels_tuple

# ---------------------- Single-patient inference for mOutViT ----------------------
def Probabilities_mOutViT(main_path,
                        num_slices,
                        cuda_id,
                        dict_out,
                        how,
                        patient_id,
                        project_name="COPD"):

    # ---------------- PATHS ----------------
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    save_path = os.path.join(main_path, f"{project_name}-results/models")
    os.makedirs(save_path, exist_ok=True)

    if num_slices == 9:
        fusion_type = "f"
    elif num_slices == 20:
        fusion_type = "w"
    else:
        raise ValueError("Invalid num_slices (must be 9 or 20).")

    # ---------------- DATASET ----------------
    class FeatureDataset(Dataset):
        def __init__(self, features_path, labels_path, dict_out):
            self.features = np.load(features_path)  # (B, num_slices, n_augmentations, last_dim)
            self.labels = np.load(labels_path)      # (B, num_tasks)

            if self.features.ndim != 4:
                raise ValueError(f"Expected features shape (B, N, A, D) but got {self.features.shape}")
            if self.labels.ndim != 2:
                raise ValueError(f"Labels should be 2D but got {self.labels.shape}")

            self.num_patients = self.features.shape[0]
            self.num_slices = self.features.shape[1]
            self.num_augmentations = self.features.shape[2]
            self.num_classes = tuple(len(v) if isinstance(v, list) else 1 for v in dict_out.values())

        def __len__(self):
            return self.num_patients

        def get_patient_data(self, patient_id):
            """Return (feature_tensor, labels_tuple) for a given patient index or ID."""
            if isinstance(patient_id, str):
                idx = self.id_to_index[patient_id]
            else:
                idx = patient_id  # assume integer index

            # Pick one random augmentation per slice
            features_list = [
                self.features[idx, slice_idx, np.random.randint(self.num_augmentations), :]
                for slice_idx in range(self.num_slices)
            ]
            feature = np.stack(features_list, axis=0)
            feature = np.expand_dims(feature, axis=1)
            feature = torch.from_numpy(feature).float()

            labels_row = self.labels[idx]
            labels_tuple = tuple(
                torch.tensor(labels_row[i]).float() if self.num_classes[i] == 1 else torch.tensor(labels_row[i]).long()
                for i in range(len(labels_row))
            )
            return feature, labels_tuple

    # ---------------- FILE PATHS ----------------
    model_id = 4
    features_dir = os.path.join(main_path, f"{project_name}{model_id}-features")
    valid_features_path = os.path.join(f"{features_dir}-{num_slices}", f"{how}_test_features.npy")
    valid_labels_path = os.path.join(f"{features_dir}-{num_slices}", f"{how}_test_labels.npy")

    valid_dataset = FeatureDataset(valid_features_path, valid_labels_path, dict_out)

    # ---------------- MODEL LOADING ----------------
    from Utils.models.mOut_transformer import mOut_transformer

    model = mOut_transformer(
        num_vectors=num_slices,
        vec_dim=1024,
        dict_out=dict_out,
        dim=256,
        depth=8,
        heads=4,
        mlp_dim=512,
        pool="cls").to(device)

    checkpoint_path = os.path.join(main_path, f"{project_name}-checkpoints","checkpoints-mOutViT",
                                   f"best_mOutViT_{model_id}{fusion_type}{how}_.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---------------- TASK SETUP ----------------
    task_names = ["EMPH", "TRAJ", "COPD"]
    num_classes_per_task = [len(v) if isinstance(v, list) else 1 for v in dict_out.values()]
    added = [0, -1, 1]

    # Compute task priors
    task_priors = []
    labels_np = np.array(valid_dataset.labels)
    for i, n_classes in enumerate(num_classes_per_task):
        add = added[i]
        task_labels = labels_np[:, i].astype(int) + add
        counts = np.bincount((task_labels - task_labels.min()).astype(int), minlength=n_classes)
        priors = counts / counts.sum()
        task_priors.append(torch.tensor(priors, dtype=torch.float32, device=device))

    # ---------------- FETCH SINGLE PATIENT ----------------
    # patient index can just be patient_id as integer index
    source_files = 'COPD-files'
    file_list_path = os.path.join(main_path, source_files, f"test-all{'_central' if num_slices == 9 else ''}.csv")
    file_list_df = pd.read_csv(file_list_path)
    patient_idx = get_patient_index_from_id(file_list_df, patient_id)
    features, labels_tuple = valid_dataset.get_patient_data(patient_idx)
    features = features.unsqueeze(0).to(device)  # batch dim
    labels_tuple = tuple(l.to(device) for l in labels_tuple)

    # ---------------- INFERENCE ----------------
    result_data = {}
    with torch.no_grad():
        outputs = model(features.squeeze(2))  # dict: task_name -> logits

        for t_idx, task_name in enumerate(task_names):
            logits = outputs[f"task{task_name}"]
            probs = torch.softmax(logits, dim=-1)
            prior = task_priors[t_idx]
            weighted_logits = logits + torch.log(prior.unsqueeze(0))
            weighted_probs = torch.softmax(weighted_logits, dim=-1)
            pred = torch.argmax(weighted_probs, dim=1)

            classes = dict_out[task_name] if isinstance(dict_out[task_name], list) else [f"class_{i}" for i in range(num_classes_per_task[t_idx])]

            # Store results
            result_data[task_name] = {
                "real": (labels_tuple[t_idx]).item(),
                "predicted": (pred.item() - added[t_idx]),
                "raw_probs": probs.cpu().numpy().flatten().tolist(),
                "weighted_probs": weighted_probs.cpu().numpy().flatten().tolist(),
                "classes": classes
            }

            # Save CSV per task
            data = [
                {"class": classes[i],
                 "raw": float(probs[0, i].cpu())}
                for i in range(num_classes_per_task[t_idx])
            ]
            data_sorted = sorted(data, key=lambda x: x["raw"], reverse=True)
            os.makedirs(f"{project_name}-results/patients", exist_ok=True)
            csv_file = os.path.join(f"{project_name}-results/patients",
                                    f"output_mOutViT_{task_name}_{fusion_type}{how}_{patient_id}.csv")
            import csv
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["class", "raw"])
                writer.writeheader()
                writer.writerows(data_sorted)

    # ---------------- PRINT RESULTS ----------------
    for task_name, task_result in result_data.items():
        print(f"\n--- {task_name} using {how} for {num_slices} slices ---")
        print(f"Patient ID: {patient_id}")
        print(f"Real label: {task_result['real']}")
        print(f"Predicted label: {task_result['predicted']}")
        print(f"Raw probabilities: {task_result['raw_probs']}")

    return result_data
