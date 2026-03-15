# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import pandas as pd

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

def balanced_accuracy_torch(outputs, labels, num_classes=None, average="micro"):
    if outputs.ndim == 2:
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = outputs
    if labels.ndim > 1:
        labels = labels.squeeze()
    correct = (preds == labels).float()
    if average == "micro":
        acc = correct.sum() / correct.numel()
    elif average == "macro":
        if num_classes is None:
            num_classes = int(labels.max().item() + 1)
        acc_per_class = []
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue  # skip classes not in batch
            acc_per_class.append(correct[mask].mean())
        acc = torch.stack(acc_per_class).mean()
    else:
        raise ValueError(f"Unknown average type: {average}")

    return acc.item()

def compute_class_weights(test_labels, num_classes, device):
    test_labels = test_labels.astype(int)
    counts = np.bincount(test_labels, minlength=num_classes)
    freqs = counts/counts.sum()
    class_weights = 1.0/freqs
    class_weights = class_weights/class_weights.sum()  # normalize to sum=1
    return torch.tensor(class_weights, dtype=torch.float32, device=device)

# ---------------- MODEL ----------------
class LearnedFusionTransformer(nn.Module):
    """
    Learns to incorporate 3 different axial embeddings via self-attention 
    to predict a clinical label.
    """
    def __init__(self, feat_dim=1024, projection_dim=512, num_classes=6, num_slices=20):
        super().__init__()
        # Project each embedding into a shared latent space
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU())
        
        # Cross-modality Attention: Learns weights for E1, E2, and E3
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=projection_dim, 
            num_heads=4, 
            batch_first=True)
        
        # Classification Head
        # We flatten the (Slices x Dim) or pool them
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * num_slices, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))

    def forward(self, e1, e2, e3):
        # Inputs shape: (B, S, D)
        B, S, D = e1.shape
        
        # Project all modalities
        p1 = self.proj(e1) # (B, S, proj_D)
        p2 = self.proj(e2)
        p3 = self.proj(e3)
        
        # Stack to learn inter-modality relationships per slice
        # Shape: (B * S, 3, proj_D)
        combined = torch.stack([p1, p2, p3], dim=2).view(B * S, 3, -1)
        
        # Attention learns "how" to incorporate the 3 embeddings
        # attn_out: (B * S, 3, proj_D)
        attn_out, weights = self.modality_attention(combined, combined, combined)
        
        # Pool the 3 modalities into 1 fused representation per slice
        fused_slices = attn_out.mean(dim=1).view(B, S, -1) # (B, S, proj_D)
        
        # Flatten slices for the final prediction
        logits = self.classifier(fused_slices.reshape(B, -1))
        
        return logits, weights

# ---------------- DATASET ----------------

class SupervisedFusionDataset(Dataset):
    """Loads 3 embeddings and their corresponding ground truth labels."""
    def __init__(self, path_e1, path_e2, path_e3, labels_path, task_idx=0):
        # Squeeze removes the augmentation dimension [B, S, 1, D] -> [B, S, D]
        self.e1 = np.load(path_e1).squeeze()
        self.e2 = np.load(path_e2).squeeze()
        self.e3 = np.load(path_e3).squeeze()
        self.labels = np.load(labels_path)
        self.task_idx = task_idx # Index for the specific label (e.g., COPD or Traj)

    def __len__(self):
        return self.e1.shape[0]

    def __getitem__(self, idx):
        # Convert to tensors
        v_e1 = torch.from_numpy(self.e1[idx]).float()
        v_e2 = torch.from_numpy(self.e2[idx]).float()
        v_e3 = torch.from_numpy(self.e3[idx]).float()
        
        label = torch.tensor(self.labels[idx, self.task_idx]).long()
        return v_e1, v_e2, v_e3, label
    
class MetaTransformerWrapper(nn.Module):
    def __init__(self, SSFM_config, wrapping_mode="cross", feat_dim=1024): 
        super().__init__()
        self.wrapping_mode = wrapping_mode
        self.feat_dim = feat_dim
        self.projection_dim = 512
        
        # Project all 3 embeddings into a shared latent space
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.ReLU())

        # Mode Selection
        if self.wrapping_mode == "gatt":
            self.modality_attention = nn.MultiheadAttention(embed_dim=self.projection_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(self.projection_dim)
            
        elif self.wrapping_mode == "cross":
            # In 'cross' mode, we use one modality to attend to others, or a joint cross-attention block
            self.cross_attn = nn.MultiheadAttention(embed_dim=self.projection_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(self.projection_dim)
            # Final projection to maintain dimensions after cross-interaction
            self.fusion_layer = nn.Linear(self.projection_dim, self.projection_dim)

        SSFM_config['vec_dim'] = self.projection_dim
        from Utils.models.transformer_LFT import transformer_LFT
        self.transformer_model = transformer_LFT(**SSFM_config)

    def forward(self, e1, e2, e3):
        B, S, D = e1.shape
        
        # 1. Project to shared space
        p1, p2, p3 = self.proj(e1), self.proj(e2), self.proj(e3)
        
        # 2. Fusion Logic
        if self.wrapping_mode == "mean":
            fused = (p1 + p2 + p3) / 3.0

        elif self.wrapping_mode == "gatt":
            # Stack per slice: (B*S, 3, proj_D)
            combined = torch.stack([p1, p2, p3], dim=2).view(B * S, 3, -1)
            attn_out, _ = self.modality_attention(combined, combined, combined)
            fused = attn_out.mean(dim=1).view(B, S, -1)
            fused = self.norm(fused)

        elif self.wrapping_mode == "cross":
            # Cross-attention: Treating modalities as a sequence of features
            # (B*S, 3, proj_D)
            combined = torch.stack([p1, p2, p3], dim=2).view(B * S, 3, -1)
            
            # Learn dependencies between Model 1, Model 2, and Model 3
            # Here combined acts as Query, Key, and Value to perform self-attention 
            # across the modality dimension.
            cross_out, _ = self.cross_attn(combined, combined, combined)
            
            # Residual connection and normalization
            fused = self.norm(combined + cross_out)
            # Reduce modality dimension to 1 (Average across fused modalities)
            fused = fused.mean(dim=1).view(B, S, -1)
            fused = self.fusion_layer(fused)

        else:
            raise ValueError(f"Unsupported wrapping_mode: {self.wrapping_mode}")

        # 3. Transformer Reasoning
        output = self.transformer_model(fused)
        return output
    
# ---------------- TRAINING FUNCTION ----------------

def Trainer_SSFM(main_path,
               cuda_id,
               project_name,
               epochs,
               batch_size,
               load_pretrained,
               load_from_checkpoint,
               lr,
               gamma,
               step,
               optzr,
               schdr,
               weight_decay,
               input_type, # Expecting 'ECG' for 3 embeddings
               output_type, # Added to handle target task (e.g., 'COPD')
               wrapping_mode): # Added to specify fusion method
    
    print("\n================ TrainerSSFM Configuration ================")
    print(f" Main path: {main_path}")
    print(f" Project: {project_name}")
    print(f" CUDA device: {cuda_id}")
    print(f" Epochs: {epochs}")
    print(f" Batch size: {batch_size}")
    print(f" Optimizer: {optzr}")
    print(f" Learning rate: {lr}")
    print(f" Which input: {input_type} (3 Embeddings)")
    print(f" Fusion Mode: {wrapping_mode}")
    print("========================================================\n")
    
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    main_checkpoint_path = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-SSFM")
    os.makedirs(main_checkpoint_path, exist_ok=True)

    # Task and Label Setup
    task_cfg = {"EMPH": 0, "TRAJ": 1, "COPD": 2, "TRAP": 3}
    IT = task_cfg[output_type]
    added = [0, -1, 1, 0] 

    # Paths (Axial only, 20 slices)
    p1_path = os.path.join(main_path, f"{project_name}1-features-20", "train_features.npy") ###E
    p2_path = os.path.join(main_path, f"{project_name}3-features-20", "train_features.npy") ###C
    p3_path = os.path.join(main_path, f"{project_name}4-features-20", "train_features.npy") ###G

    l_path = os.path.join(main_path, f"{project_name}3-features-20", "train_labels.npy") ###C for COPD

    # Dataset & Dataloader
    class SSLFeatureDataset(Dataset):
        def __init__(self, p1, p2, p3, lp):
            self.e1, self.e2, self.e3 = np.load(p1).squeeze(), np.load(p2).squeeze(), np.load(p3).squeeze()
            self.labels = np.load(lp)
        def __len__(self): return self.e1.shape[0]
        def __getitem__(self, idx):
            return torch.from_numpy(self.e1[idx]).float(), \
                   torch.from_numpy(self.e2[idx]).float(), \
                   torch.from_numpy(self.e3[idx]).float(), \
                   torch.tensor(self.labels[idx])

    full_dataset = SSLFeatureDataset(p1_path, p2_path, p3_path, l_path)
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class Weights for target task
    raw_labels = np.load(l_path)[:, IT].squeeze() + added[IT]
    if output_type == "COPD": raw_labels[raw_labels == -1] = 0
    num_classes = len(np.unique(raw_labels))
    task_class_weights = compute_class_weights(raw_labels, num_classes, device)

    # Model Configuration
    SSFM_config = {
        'num_vectors': 20,
        'vec_dim': 512,
        'num_classes': num_classes,
        'dim': 512,
        'depth': 8,
        'heads': 8,
        'mlp_dim': 1024,
        'pool': "cls"}

    model = MetaTransformerWrapper(SSFM_config, wrapping_mode=wrapping_mode).to(device)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if optzr == "adam" else \
                optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step)
    criterion = nn.CrossEntropyLoss(weight=task_class_weights)
    
    # ---------------- LOADING & INITIALIZATION ----------------
    # Updated path logic to remove vars_add0 but keep your specific naming structure
    if load_pretrained == True and load_from_checkpoint == False:
        checkpoint_path = os.path.join(main_path, f"{project_name}-models/model_SSFM_{output_type}-{input_type}_{wrapping_mode}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        last_epoch = 0
        best_val_acc = 0.0
        print(f"Loaded pretrained weights from {checkpoint_path}\n")

    elif load_from_checkpoint == True and load_pretrained == False:
        checkpoint_path = os.path.join(main_checkpoint_path, f"best_SSFM_{output_type}_{input_type}_{wrapping_mode}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device=device)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_val_acc = checkpoint.get('val_acc', 0.0)
        last_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded best model from epoch {last_epoch}")

    elif load_from_checkpoint == False and load_pretrained == False:
        model = model.to(device=device)
        last_epoch = 0
        best_val_acc = 0.0
        print("Starting training from scratch")

    else:
        raise ValueError("You cannot set both 'load_pretrained' and 'load_from_checkpoint' to the same value.")

    # ---------------- TRAINING ----------------
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    
    best_epoch = last_epoch
    epochs_no_improve = 0
    patience = 20
    best_val_loss = float('inf')

    criterion = nn.CrossEntropyLoss(weight=task_class_weights)
    
    for epoch in range(last_epoch, last_epoch + epochs):
        model.train()
        total_loss, total_acc = 0, 0
        optimizer.zero_grad()

        for e1, e2, e3, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            # Move all 3 embeddings to device
            e1, e2, e3 = e1.to(device), e2.to(device), e3.to(device)
            
            # Target handling (Supervised label for the SSL-fusion task)
            labels = labels[:, IT].squeeze().to(device).long() + added[IT]
            if output_type == "COPD":
                labels[labels == -1] = 0
            
            # Model forward pass with 3 embeddings
            outputs = model(e1, e2, e3)
            
            loss = criterion(outputs, labels)
            acc = balanced_accuracy_torch(outputs, labels, average="macro")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_acc)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_acc = 0, 0

        with torch.no_grad():
            for e1, e2, e3, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                e1, e2, e3 = e1.to(device), e2.to(device), e3.to(device)
                
                labels = labels[:, IT].squeeze().to(device).long() + added[IT]
                if output_type == "COPD":
                    labels[labels == -1] = 0
                    
                outputs = model(e1, e2, e3)
                loss = criterion(outputs, labels)
                acc = balanced_accuracy_torch(outputs, labels, average="macro")
                    
                val_loss += loss.item()
                val_acc += acc

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_acc = val_acc / len(valid_loader)
        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_acc)

        # Scheduler step
        if schdr == "plat":
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train loss={avg_train_loss:.4f}, acc={avg_train_acc:.4f} | "
            f"Val loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}")
        
        # ---------------- SAVING & CHECKPOINTING ----------------
        improved_acc = avg_val_acc > best_val_acc + 1e-4
        improved_loss = avg_val_loss < best_val_loss - 1e-4

        if improved_acc or improved_loss:
            best_val_acc = max(best_val_acc, avg_val_acc)
            best_val_loss = min(best_val_loss, avg_val_loss)
            best_epoch = epoch + 1
            epochs_no_improve = 0

            # Path includes fusion type (how), input models (ETC), and wrapping mode (cross/gatt)
            best_model_path = os.path.join(main_checkpoint_path, f"best_SSFM_{output_type}-{input_type}_{wrapping_mode}.pt")
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc}, best_model_path)

            print(f"Best model updated at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")
        
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} (val_acc: {best_val_acc:.4f})")
            break
        
    # Save final model
    final_model_path = os.path.join(main_path, f"{project_name}-models/model_SSFM_{output_type}-{input_type}_{wrapping_mode}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": validation_loss[-1]}, final_model_path)

    print("\nTraining complete.")
    print(f"Saved final model at: {final_model_path}")
    print(f"Best SSFM using {input_type} for {output_type} model: and wrapping {wrapping_mode} saved at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")

    return {"train_loss": training_loss, "val_loss": validation_loss, "train_acc": training_acc, "val_acc": validation_acc}