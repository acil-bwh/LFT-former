# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
#       ···················· BWH  ····················
# ··························································

from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm

# ---------------- SEEDING ----------------
def seed_everything(seed=3):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(3)

def compute_class_weights(test_labels, num_classes, device):
    test_labels = test_labels.astype(int)
    counts = np.bincount(test_labels, minlength=num_classes)
    counts = np.maximum(counts, 1) 
    freqs = counts/counts.sum()
    weights = 1.0/freqs
    weights = weights*(num_classes/weights.sum())
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ---------------- TRAINING FUNCTION ----------------
def Trainer_augViT(main_path,
               model_id,
               num_slices,
               cuda_id,
               project_name,
               epochs,
               batch_size,
               load_pretrained,
               load_from_checkpoint,
               lr,
               gamma,
               step,
               weight_decay):
    """
    Train a multi-slice ViT model with patient-level features.

    Args:
        main_path (str): Root directory path.
        model_id (int): 1=Emphysema, 2=Trajectories, 3=COPD.
        num_slices (int): Slices per patient (e.g., 9 or 20).
        cuda_id (int): GPU device ID.
        project_name (str): Project prefix.
        epochs (int): Training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        weight_decay (float): L2 weight decay.
        gamma (float): LR scheduler decay.
        load_pretrained (bool): Whether to resume from checkpoint.
        start_epoch (int): Starting epoch if resuming.
        group_by_patient (bool): Group slices into patients automatically.
    """

    # ---------------- Summary ----------------
    print("\n================ Training Configuration ================")
    print(f" Main path: {main_path}")
    print(f" Project: {project_name}")
    print(f" Model ID: {model_id}")
    print(f" Num slices: {num_slices}")
    print(f" CUDA device: {cuda_id}")
    print(f" Epochs: {epochs}")
    print(f" Batch size: {batch_size}")
    print(f" Pretrained: {load_pretrained}")
    print(f" Resume checkpoint: {load_from_checkpoint}")
    print("========================================================\n")

    # ---------------- DEVICE ----------------
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # ---------------- PATHS ----------------
    main_checkpoint_path = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-augViT")
    os.makedirs(main_checkpoint_path, exist_ok=True)

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
    elif model_id == 4:
        print(f"···· TRAPPING MODEL {num_slices}-slices ····")
        num_classes = 4
        add = 0
    elif model_id == 5:
        print(f"···· COPD-5 MODEL {num_slices}-slices ····")
        num_classes = 5
        add = 0
    elif model_id == 6:
        print(f"···· COPD-2 MODEL {num_slices}-slices ····")
        num_classes = 2
        add = 0
    else:
        raise ValueError("Invalid model_id (must be 2, or 3).")

    if num_slices == 9:
        fusion_type = "C"
    elif num_slices == 20:
        fusion_type = "W"
    else:
        raise ValueError("num_slices must be 9 or 20.")

    class FeatureDataset(Dataset):
        def __init__(self, features_path, labels_path, model_id):
            self.features = np.load(features_path)  # (B, num_slices, n_augmentations, last_dim)
            self.labels = np.load(labels_path)      # (B,4)
            if model_id in [3,5,6]:
                which = 2
            else:
                which = model_id-1

            self.labels = self.labels[:,which]

            self.model_id = model_id

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

            aug_idx = np.random.randint(self.num_augmentations, size=self.num_slices)  # shape (num_slices,)
            feature = self.features[idx, np.arange(self.num_slices), aug_idx, :]       # (num_slices, last_dim)

            feature = np.expand_dims(feature, axis=1)

            feature = torch.from_numpy(feature).float()
            label = self.labels
            label = torch.tensor(label[idx]).long()

            return feature, label
        
    source_features = f"{project_name}-embeddings/{project_name}{model_id}-features-{num_slices}"
    features_dir = os.path.join(main_path, source_features)
    
    features_path = os.path.join(features_dir, "train_features.npy")
    labels_path = os.path.join(features_dir, f"train_labels.npy")

    train_dataset = FeatureDataset(features_path, labels_path, model_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # batch of patients

    valid_features_path = os.path.join(features_dir, "valid_features.npy")
    valid_labels_path = os.path.join(features_dir, f"valid_labels.npy")

    valid_dataset = FeatureDataset(valid_features_path, valid_labels_path, model_id)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # ---------------- MODEL ----------------
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

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=gamma, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer,
    # mode='min',
    # factor=0.9,
    # patience=10,
    # min_lr=1e-9)

    labels = np.load(labels_path)
    if model_id in [3,5,6]:
        which = 2
    else:
        which = model_id-1

    labels = labels[:,which].squeeze()
    labels[labels == -2] = 0

    if model_id == 5:
        labels[labels==-1]=0
    
    elif model_id == 6:
        labels[labels==-1]=0
        labels[labels>0]=1

    labels = labels + add

    unique_values, counts = np.unique(labels, return_counts=True)
    print(f"Total Unique Values: {len(unique_values)}")
    print(f"Unique values: ", unique_values)
    for val, count in zip(unique_values, counts):
        print(f"Label {round(val)}: {count} occurrences")

    num_classes = len(unique_values)
    task_class_weights = compute_class_weights(labels, num_classes, device)
    print("Weights per class: ", task_class_weights)
    print("Shape of weights: ", task_class_weights.shape)

    criterion = nn.CrossEntropyLoss(weight=task_class_weights)

    if load_pretrained == True and load_from_checkpoint == False:
        checkpoint_path = os.path.join(main_path, f"{project_name}-models/model_augViT_{project_name}_{model_id}{fusion_type}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        last_epoch = 0
        best_val_acc = 0.0
        print(f"Loaded pretrained weights from {checkpoint_path}\n")

    elif load_from_checkpoint == True and load_pretrained == False:
        checkpoint_path = os.path.join(main_checkpoint_path, f"best_augViT_{project_name}_{model_id}{fusion_type}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device=device)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_val_acc = checkpoint.get('val_acc', 0.0)  # load the best validation accuracy
        last_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded best model from epoch {last_epoch}")

    elif load_from_checkpoint == False and load_pretrained == False:
        model = model.to(device=device)
        last_epoch = 0
        best_val_acc = 0.0
        print("Starting training from scratch")

    else:
        print(f"load_pretrained is set to: {load_pretrained}")
        print(f"load_from_checkpoint is set to: {load_from_checkpoint}")
        raise ValueError("You cannot set both 'load_pretrained' and 'load_from_checkpoint' to the same value.")

    # ---------------- TRAINING ----------------
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    
    print(f"Training for {epochs} epochs...\n")

    best_epoch = last_epoch
    epochs_no_improve = 0
    patience = 50
    best_val_loss = float('inf')

    for epoch in range(last_epoch, last_epoch + epochs):
        model.train()
        total_loss, total_acc = 0, 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            features, labels = features.to(device), labels.to(device)

            labels[labels == -2] = 0
    
            if model_id == 5:
                labels[labels==-1]=0
            
            elif model_id == 6:
                labels[labels==-1]=0
                labels[labels>0]=1

            labels = labels + add

            optimizer.zero_grad()

            outputs = model(features.squeeze(2))
            loss = criterion(outputs, labels)
            acc = multiclass_accuracy(outputs, labels, num_classes=num_classes, average="micro")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_acc)

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for features, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                features, labels = features.to(device), labels.to(device)
                
                labels[labels == -2] = 0
    
                if model_id == 5:
                    labels[labels==-1]=0
                
                elif model_id == 6:
                    labels[labels==-1]=0
                    labels[labels>0]=1

                labels = labels + add

                outputs = model(features.squeeze(2))

                loss = criterion(outputs, labels)
                acc = multiclass_accuracy(outputs, labels, num_classes=num_classes, average="micro")

                val_loss += loss.item()
                val_acc += acc.item()

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_acc = val_acc / len(valid_loader)
        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_acc)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train loss={avg_train_loss:.4f}, acc={avg_train_acc:.4f} | "
            f"Val loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}")

        improved_acc = avg_val_acc > best_val_acc + 1e-4
        improved_loss = avg_val_loss < best_val_loss - 1e-4

        if improved_acc or improved_loss:
            best_val_acc = max(best_val_acc, avg_val_acc)
            best_val_loss = min(best_val_loss, avg_val_loss)
            best_epoch = epoch + 1
            epochs_no_improve = 0

            best_model_path = os.path.join(main_checkpoint_path, f"best_augViT_{project_name}_{model_id}{fusion_type}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
                "val_acc": best_val_acc}, best_model_path)
            print(f"Best model updated at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")
        
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} (val_acc: {best_val_acc:.4f})")
            break
        
    final_model_path = os.path.join(main_path, f"{project_name}-models/model_augViT_{project_name}_{model_id}{fusion_type}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": validation_loss[-1]}, final_model_path)

    print("\nTraining complete.")
    print(f"Saved final model at: {final_model_path}")
    print(f"Best model saved at epoch {last_epoch+best_epoch} with val_acc: {best_val_acc:.4f}")

    return {"train_loss": training_loss,
        "val_loss": validation_loss,
        "train_acc": training_acc,
        "val_acc": validation_acc}