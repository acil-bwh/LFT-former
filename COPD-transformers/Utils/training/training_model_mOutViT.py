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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

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

import torch

def balanced_accuracy_torch(outputs, labels, num_classes=None, average="micro"):
    """
    Compute multi-class accuracy for classification outputs.

    Args:
        outputs (torch.Tensor): Model outputs, shape [B, C] (logits or probabilities)
        labels (torch.Tensor): True labels, shape [B] (class indices)
        num_classes (int, optional): Number of classes
        average (str): "micro" or "macro"

    Returns:
        float: Accuracy value
    """
    # Convert logits to predicted class indices
    if outputs.ndim == 2:
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = outputs  # already class indices

    # Ensure labels are 1D
    if labels.ndim > 1:
        labels = labels.squeeze()

    correct = (preds == labels).float()

    if average == "micro":
        # overall accuracy
        acc = correct.sum() / correct.numel()
    elif average == "macro":
        # per-class accuracy
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

    return acc.item()  # convert to Python float

def r_squared_torch(outputs, targets):
    outputs = outputs.view(-1).float()
    targets = targets.view(-1).float()
    y_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - y_mean) ** 2)
    ss_res = torch.sum((targets - outputs) ** 2)
    if ss_tot < 1e-8:
        return torch.tensor(1.0, device=outputs.device)
    r2 = 1.0 - (ss_res / ss_tot)
    return r2

def get_task_weights(full_dataset, train_indices, task_idx, details, device):
    num_classes = details["classes"]
    add = details["adjustment"]
    
    labels = full_dataset.labels[train_indices, task_idx].astype(int)
    labels = labels+add
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0/(counts) 
    weights = weights / weights.sum() * num_classes 

    return torch.tensor(weights, dtype=torch.float32).to(device)

def compute_mse_torch(output, target):
    return F.mse_loss(output.squeeze(-1), target)

class FeatureDataset(Dataset):
    def __init__(self, features_path, csv_path, dict_out, normalization):

        self.normalization = normalization
        self.path = csv_path
        df = pd.read_csv(csv_path)
        self.age = df["age_visit_P1"].values.astype(np.float32)
        self.gender = df["gender_P1"].values.astype(np.float32)
        self.bmi = df["BMI_P1"].values.astype(np.float32)
        self.race = df["race_P1"].values.astype(np.float32)
        
        if self.normalization:
            mean_age = self.age.mean()
            std_age = self.age.std()
            mean_bmi = self.bmi.mean()
            std_bmi = self.bmi.std()            
            self.age_processed = (self.age - mean_age) / (std_age + 1e-6)
            self.bmi_processed = (self.bmi - mean_bmi) / (std_bmi + 1e-6)
            self.gender_processed = self.gender - 1.0
            self.race_processed = self.race - 1.0

        self.task_columns_ordered = list(dict_out.keys())
        self.labels = df[self.task_columns_ordered].values.astype(np.float32)
        self.num_classes = tuple(len(v) if isinstance(v, list) else 1 for v in dict_out.values())

        self._process_all_labels()

        self.features = np.load(features_path) 
        
        if self.features.ndim != 4:
            raise ValueError(f"Expected (B, S, A, D), got {self.features.shape}")

    def _process_all_labels(self):
        for idx, col in enumerate(self.task_columns_ordered):
            data = self.labels[:, idx]

            if col == "finalgold_visit_P1":
                self.labels[:, idx] = np.where(data == -2, 0, data)

            elif col == "eosinphl_P2":
                self.labels[:, idx] = np.digitize(data, [0.150, 0.300])

            elif col == "Pi10_Thirona_P1":
                self.labels[:, idx] = np.digitize(data, [3.3, 3.6, 4.0])

            elif col == "pctEmph_Thirona_P1":
                self.labels[:, idx] = np.digitize(data, [10.0, 20.0, 30.0])

            elif col == "FEV1pp_post_P1":
                self.labels[:, idx] = np.digitize(data, [30.0, 50.0, 65.0, 80.0])

            elif col == "lung_density_vnb_P1":
                self.labels[:, idx] = np.digitize(data, [50.0, 80.0, 100.0])

            elif col == "distwalked_P1":
                self.labels[:, idx] = np.digitize(data, [500, 800, 1200])

            elif col == "ChangeFEV1pp_P1P2":
                self.labels[:, idx] = np.digitize(data, [-20, 0, 20, 40])

            elif col == "ChangeVNB_LD_P1P2":
                self.labels[:, idx] = np.digitize(data, [-10, 10, 30])

            elif col == "Changedistwalked_P1P2":
                self.labels[:, idx] = np.digitize(data, [-500, 0, 500, 800])

            elif col == "ChangeVA_LD_P1P2":
                self.labels[:, idx] = np.digitize(data, [-50, 0, 50])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        num_slices = self.features.shape[1]
        num_views = self.features.shape[2]
        aug_indices = np.random.randint(num_views, size=num_slices)
        
        feature = self.features[idx, np.arange(num_slices), aug_indices, :]
        feature = torch.from_numpy(feature).float() # (S, D)
        
        if self.normalization == True:
            age_val = self.age_processed[idx]
            gen_val = self.gender_processed[idx]
            bmi_val = self.bmi_processed[idx]
            race_val = self.race_processed[idx]
            meta = torch.tensor([age_val, gen_val, race_val, bmi_val], dtype=torch.float32)
            meta = meta.repeat(num_slices, 1) # (S, 2)
            combined_feature = torch.cat([feature, meta], dim=-1)
        else:
            combined_feature = feature

        labels_row = self.labels[idx]
        labels_list = []
        for i in range(len(labels_row)):
            is_regression = self.num_classes[i] == 1
            if is_regression:
                labels_list.append(torch.tensor(labels_row[i]).float())
            else:
                labels_list.append(torch.tensor(labels_row[i]).long())

        return combined_feature, tuple(labels_list)

# ---------------- TRAINING FUNCTION ----------------
def Trainer_mOutViT(main_path,
               num_slices,
               cuda_id,
               project_name,
               epochs,
               batch_size,
               dict_out,
               task_dict,
               input_type,
               load_pretrained,
               how,
               load_from_checkpoint,
               lr,
               gamma,
               step,
               weight_decay):

    print("\n================ Training Configuration ================")
    print(f" Main path: {main_path}")
    print(f" Project: {project_name}")
    print(f" Num slices: {num_slices}")
    print(f" CUDA device: {cuda_id}")
    print(f" Epochs: {epochs}")
    print(f" Batch size: {batch_size}")
    print(f" Input type - embeddings: {input_type}")
    print(f" Fusion of weights (if any): {how}")
    print(f" Output type - tasks: {dict_out.keys()}")
    print(f" Pretrained: {load_pretrained}")
    print(f" Resume checkpoint: {load_from_checkpoint}")
    print("========================================================\n")

    # ---------------- DEVICE ----------------
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # ---------------- PATHS ----------------
    main_checkpoint_path = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-mOutViT")
    os.makedirs(main_checkpoint_path, exist_ok=True)

    if num_slices == 9:
        fusion_type = "C"
    elif num_slices == 20:
        fusion_type = "W"
    else:
        raise ValueError("num_slices must be 9 or 20.")

    ### dim adding for age and/or gender and/or pack-years...

    feat_dim = 1024
    vars_add = 0

    if len(input_type) != 1:
        if how == "concat":
            vec_dim = feat_dim*len(input_type)+vars_add
        elif how == "mix":
            vec_dim = feat_dim*(len(input_type)-1)+vars_add
        else:
            vec_dim = feat_dim+vars_add
    else:
        vec_dim = feat_dim+vars_add

    model_name0 = "copd"
    features_dir = os.path.join(main_path, f"{project_name}-features")
    
    from torch.utils.data import random_split
    print("\Full dataset loading...")
    tr_path = os.path.join(main_path, f"{project_name}-files",f"df_train_{model_name0}.csv")
    train_features_path = os.path.join(f"{features_dir}-{num_slices}", f"{how}_train_features_{input_type}.npy")

    full_dataset = FeatureDataset(train_features_path, tr_path, dict_out, normalization=False)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset_split, val_dataset_split = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_indices = train_dataset_split.indices
    
    print("\nApplying normalization based on training set statistics...")    
    training_norm_params = {}
    for i, col_name in enumerate(full_dataset.task_columns_ordered):
        if task_dict[col_name]["type"] == "regression":
            train_data = full_dataset.labels[train_indices, i]
            mask = ~np.isnan(train_data)
            training_norm_params[i] = {
                'mean': torch.tensor(np.nanmean(train_data[mask]), device=device),
                'std': torch.tensor(np.nanstd(train_data[mask]) + 1e-8, device=device)}
    
    print("\nTraining: loading data...")
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    print("\nValidation: loading data...")
    valid_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=True)
    norm_params = training_norm_params

    # ---------------- MODEL ----------------
    from Utils.models.mOut_transformer import mOut_transformer
    model = mOut_transformer(
        num_vectors=num_slices,
        vec_dim=vec_dim,
        dict_out=dict_out,
        dim=256,
        depth=8,
        heads=4,
        mlp_dim=512,
        pool="cls")

    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

    if load_pretrained == True and load_from_checkpoint == False:
        checkpoint_path = os.path.join(main_path, f"{project_name}-models/model_mOutViT_{fusion_type}-{how}_{input_type}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device=device)
        last_epoch = 0
        best_val_acc = 0.0
        print(f"\nLoaded pretrained weights from {checkpoint_path}\n")

    elif load_from_checkpoint == True and load_pretrained == False:
        checkpoint_path = os.path.join(main_checkpoint_path, f"best_mOutViT_{fusion_type}-{how}_{input_type}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device=device)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        best_val_acc = checkpoint.get('val_acc', 0.0)  # load the best validation accuracy
        last_epoch = checkpoint.get('epoch', 0)
        print(f"\nLoaded best model from epoch {last_epoch}")

    elif load_from_checkpoint == False and load_pretrained == False:
        model = model.to(device=device)
        last_epoch = 0
        best_val_acc = 0.0
        print("\nStarting training from scratch")

    else:
        print(f"load_pretrained is set to: {load_pretrained}")
        print(f"load_from_checkpoint is set to: {load_from_checkpoint}")
        raise ValueError("You cannot set both 'load_pretrained' and 'load_from_checkpoint' to the same value.")

    # ---------------- TRAINING ----------------
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    
    print(f"Training mOutViT with weights from {input_type} for {epochs} epochs...\n")

    best_epoch = last_epoch
    epochs_no_improve = 0
    patience = 20
    best_val_acc = 0.0
    best_val_loss = float('inf')

    task_config = []
    label_column_index = 0

    for i, (name, details) in enumerate(task_dict.items()): # name is "EMPH", "COPD", etc.
        if details["type"] == "regression":
            criterion_func = nn.MSELoss()
        else:
            cls_weights = get_task_weights(full_dataset, train_indices, i, details, device)
            # criterion_func = nn.CrossEntropyLoss(weight=cls_weights)
            criterion_func = nn.CrossEntropyLoss()

        task_config.append({
            "name": name,
            "index": label_column_index,
            "output": f"task{name}",
            "criterion": criterion_func,
            "weight": details["weight"],
            "adjustment": details["adjustment"],
            "is_regression": (details["type"] == "regression"),
            "num_classes": details["classes"]})
        
        label_column_index += 1
        
    total_weights = sum(t['weight'] for t in task_config)
    task_names = [t['name'] for t in task_config]
    output_num = len(task_config)
    
    for epoch in range(last_epoch, last_epoch + epochs):
        model.train()
        total_loss, total_acc = 0, 0
        tr_loss_task = [0.0] * output_num
        tr_acc_task = [0.0] * output_num

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            input_features = features.squeeze(2).to(device)
            labels = [l.to(device) for l in labels]
            optimizer.zero_grad()
            outputs = model(input_features)
            
            batch_weighted_loss = 0
            batch_total_acc = 0

            for i, config in enumerate(task_config):
                raw_labels = labels[config['index']]
                task_outputs = outputs[config['output']]
                
                nan_mask = ~torch.isnan(raw_labels)
                filtered_labels = raw_labels[nan_mask]
                filtered_outputs = task_outputs[nan_mask]

                if filtered_labels.shape[0] == 0:
                    continue
                                    
                task_labels = filtered_labels + config['adjustment']
                
                if config['is_regression']:
                    task_labels_unsq = task_labels.unsqueeze(1)
                    
                    mean_t = torch.tensor(norm_params[config['index']]['mean'], device=device)
                    std_t = torch.tensor(norm_params[config['index']]['std'], device=device)
                    
                    pred_unscaled = filtered_outputs*std_t+mean_t
                    true_unscaled = task_labels_unsq*std_t+mean_t
                    
                    task_metric = r_squared_torch(pred_unscaled, true_unscaled)
                    task_loss = config['criterion'](filtered_outputs, task_labels_unsq)

                else:
                    task_labels_int = task_labels.long()
                    task_metric = balanced_accuracy_torch(filtered_outputs, task_labels_int)
                    task_loss = config['criterion'](filtered_outputs, task_labels_int)

                batch_weighted_loss += config['weight']*task_loss
                batch_total_acc += task_metric
                
                tr_loss_task[i] += task_loss.item()
                tr_acc_task[i] += task_metric

            combined_loss = batch_weighted_loss/total_weights
            avg_batch_metric = batch_total_acc/output_num

            combined_loss.backward()
            optimizer.step()

            total_loss += combined_loss.item()
            total_acc += avg_batch_metric

        avg_train_loss = total_loss/len(train_loader)
        avg_train_metric = total_acc/len(train_loader)
        avg_loss_task = [l/len(train_loader) for l in tr_loss_task]
        avg_acc_task = [a/len(train_loader) for a in tr_acc_task]
        avg_rmse_task = [np.sqrt(l) if task_config[i]['is_regression'] else 0.0 
                         for i, l in enumerate(avg_loss_task)]
        
        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_metric)

        model.eval()
        val_loss, val_acc = 0, 0
        val_loss_task = [0.0]*output_num
        val_acc_task = [0.0]*output_num
        
        with torch.no_grad():
            for features, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_features = features.squeeze(2).to(device)
                labels = [l.to(device) for l in labels]
                outputs = model(input_features)
                batch_weighted_loss = 0
                batch_total_acc = 0

                for i, config in enumerate(task_config):
                    raw_labels = labels[config['index']]
                    task_outputs = outputs[config['output']]
                    
                    nan_mask = ~torch.isnan(raw_labels)
                    filtered_labels = raw_labels[nan_mask]
                    filtered_outputs = task_outputs[nan_mask]

                    if filtered_labels.shape[0] == 0:
                        continue
                                        
                    task_labels = filtered_labels + config['adjustment']
                    
                    if config['is_regression']:
                        task_labels_unsq = task_labels.unsqueeze(1)
                        
                        mean_t = torch.tensor(norm_params[config['index']]['mean'], device=device)
                        std_t = torch.tensor(norm_params[config['index']]['std'], device=device)
                        
                        pred_unscaled = filtered_outputs * std_t + mean_t
                        true_unscaled = task_labels_unsq * std_t + mean_t
                        
                        task_metric = r_squared_torch(pred_unscaled, true_unscaled)
                        task_loss = config['criterion'](filtered_outputs, task_labels_unsq)

                    else:
                        task_labels_int = task_labels.long()
                        task_metric = balanced_accuracy_torch(filtered_outputs, task_labels_int)
                        task_loss = config['criterion'](filtered_outputs, task_labels_int)

                    batch_weighted_loss += config['weight'] * task_loss
                    batch_total_acc += task_metric
                    val_loss_task[i] += task_loss.item()
                    val_acc_task[i] += task_metric
                
                combined_loss = batch_weighted_loss / total_weights
                val_loss += combined_loss.item()
                val_acc += (batch_total_acc / output_num)

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_metric = val_acc / len(valid_loader)
        avg_val_loss_task = [l / len(valid_loader) for l in val_loss_task]
        avg_val_acc_task = [a / len(valid_loader) for a in val_acc_task]
        avg_val_rmse_task = [np.sqrt(l) if task_config[i]['is_regression'] else 0.0 
                             for i, l in enumerate(avg_val_loss_task)]

        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_metric)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss) 
        else:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train loss={avg_train_loss:.4f}, metric={avg_train_metric:.4f} | "
            f"Val loss={avg_val_loss:.4f}, metric={avg_val_metric:.4f}")
        print("—" * 100)

        print(f"Epoch {epoch+1} TRAINING Summary:")
        train_loss_str = ", ".join([f"{name}={loss:.4f}" for name, loss in zip(task_names, avg_loss_task)])
        train_acc_str = ", ".join([f"{name}={acc:.4f}" for name, acc in zip(task_names, avg_acc_task)])
        train_rmse_str = ", ".join([f"{name}={rmse:.4f}" for name, rmse in zip(task_names, avg_rmse_task) 
                                    if task_config[task_names.index(name)]['is_regression']])
        
        print(f"Task-wise Loss: {train_loss_str}")
        print(f"Task-wise Metric: {train_acc_str}")
        if train_rmse_str:
            print(f"Task-wise RMSE (Normalized): {train_rmse_str}")

        print(f"\nEpoch {epoch+1} VALIDATION Summary:")
        val_loss_str = ", ".join([f"{name}={loss:.4f}" for name, loss in zip(task_names, avg_val_loss_task)])
        val_acc_str = ", ".join([f"{name}={acc:.4f}" for name, acc in zip(task_names, avg_val_acc_task)])
        val_rmse_items = [f"{task_names[i]}={avg_val_rmse_task[i]:.4f}" 
                          for i in range(output_num) if task_config[i]['is_regression']]
        val_rmse_str = ", ".join(val_rmse_items)

        print(f"Task-wise Loss: {val_loss_str}")
        print(f"Task-wise Metric (R2/Acc): {val_acc_str}")
        if val_rmse_str:
            print(f"Task-wise RMSE: {val_rmse_str}")

        print("—" * 100)

        improved_acc = avg_val_metric > best_val_acc + 1e-4
        improved_loss = avg_val_loss < best_val_loss - 1e-4

        if improved_acc or improved_loss:
            best_val_acc = max(best_val_acc, avg_val_metric)
            best_val_loss = min(best_val_loss, avg_val_loss)

            best_epoch = epoch + 1
            epochs_no_improve = 0

            best_model_path = os.path.join(main_checkpoint_path, f"best_mOutViT_{fusion_type}-{how}_{input_type}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
                "metric": best_val_acc}, best_model_path)
            
            print(f"Best model updated at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")
        
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} (val_acc: {best_val_acc:.4f})")
            break
        
    final_model_path = os.path.join(main_path, f"{project_name}-models/model_mOutViT_{fusion_type}-{how}_{input_type}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": validation_loss[-1]}, final_model_path)

    print("\nTraining complete.")
    print(f"Saved final model at: {final_model_path}")
    print(f"Best mOutViT model saved at epoch {last_epoch+best_epoch} with val_acc: {best_val_acc:.4f}")

    return {"train_loss": training_loss,
        "val_loss": validation_loss,
        "train_acc": training_acc,
        "val_acc": validation_acc}