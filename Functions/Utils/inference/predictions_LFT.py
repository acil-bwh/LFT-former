# ··························································
# ············ Applied Chest Imaging Lab, 2026 ·············
# ·········· @acil-bwh | @queraltmartinsaladich ············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score

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

def compute_class_weights(labels, num_classes):
    """
    labels: numpy array of shape (num_samples,) with integer class labels
    num_classes: int, total number of classes for the task
    returns: tensor of class weights (inverse frequency)
    """
    counts = np.bincount(labels, minlength=num_classes)
    freqs = counts/counts.sum()
    class_weights = 1.0/freqs
    class_weights = class_weights/class_weights.sum()  # normalize to sum=1
    return torch.tensor(class_weights, dtype=torch.float32)

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

# ---------------- PREDICTOR FUNCTION ----------------
def Predictor_LFT(main_path,
                num_slices,
                cuda_id,
                project_name,
                vars_add,
                wrapping_mode):
        
    # ----------- PATHS & DEVICE -----------
    cuda_set = f"cuda:{cuda_id}"

    device = torch.device(cuda_set if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    added = -1

    # ----------- DATASET -----------

    features_dir = os.path.join(main_path, f"{project_name}-embeddings/{project_name}-features")
    project_name_dir = project_name
    csv_path = os.path.join(main_path, f"{project_name_dir}-files",f"df_val_traj.csv")

    models_path = os.path.join(main_path, f"{project_name_dir}-checkpoints/checkpoints-LFT/")
    save_path = os.path.join(main_path, f"{project_name}-results/models")
    os.makedirs(save_path, exist_ok=True)

    valid_features_path = os.path.join(f"{features_dir}", f"test_features.npy")
    valid_labels_path = os.path.join(f"{features_dir}", f"test_labels.npy")
    
    # ----------- MODEL LOADING -----------
    class FeatureDataset(Dataset):
        def __init__(self, features_path, labels_path, csv_path, vars_add0, stats=None):
            self.features = np.load(features_path)      # (B, slices, aug, dim)
            self.labels = np.load(labels_path)          # (B, num_tasks)
            B = self.features.shape[0]
            self.vars_add0 = vars_add0
            
            df = pd.read_csv(csv_path)
            if "sid" not in df.columns:
                print("WARNING: Column 0 is not named 'sid'. Using column 0 anyway.")
                df.rename(columns={df.columns[0]: "sid"}, inplace=True)

            if self.vars_add0 != 0:
                required_cols = ["age", "gender", "packs", "emph", "race", "bmi"]
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"CSV missing required column: {col}")
                    
                self.col_A = df["age"].values.astype(float)
                self.col_B = df["gender"].values.astype(float)
                self.col_C = df["packs"].values.astype(float)
                self.col_D = df["emph"].values.astype(float)
                self.col_E = df["race"].values.astype(float)
                self.col_F = df["bmi"].values.astype(float)

                self.age_mean = self.col_A.mean()
                self.age_std  = self.col_A.std() + 1e-6
                
                self.emph_mean = self.col_D.mean()
                self.emph_std = self.col_D.std() + 1e-6
                
                packs_log_all = np.log1p(self.col_C)
                self.packs_log_mean = packs_log_all.mean()
                self.packs_log_std = packs_log_all.std() + 1e-6

                self.packs_mean = self.col_C.mean()
                self.packs_std = self.col_C.std() + 1e-6

                self.bmi_mean = self.col_F.mean()
                self.bmi_std = self.col_F.std() + 1e-6

                if stats is None:
                    self.stats = {
                        'age_mean': self.col_A.mean(),
                        'age_std': self.col_A.std() + 1e-6,
                        'emph_mean': self.col_D.mean(),
                        'emph_std': self.col_D.std() + 1e-6,
                        'packs_log_mean': np.log1p(self.col_C).mean(),
                        'packs_log_std': np.log1p(self.col_C).std() + 1e-6,
                        'bmi_mean': self.col_F.mean(),
                        'bmi_std': self.col_F.std() + 1e-6}
                else:
                    self.stats = stats

            self.num_patients = B
            self.num_slices = self.features.shape[1]
            self.num_augmentations = self.features.shape[2]

        def __len__(self):
            return self.num_patients

        def __getitem__(self, idx):

            aug_idx = np.random.randint(self.num_augmentations, size=self.num_slices)
            feat = self.features[idx, np.arange(self.num_slices), aug_idx, :]
            feat = torch.from_numpy(feat[:, None, :]).float()
            
            labels = self.labels[idx]

            if self.vars_add0 != 0:
                gender_norm = float(self.col_B[idx]) - 1.0
                race_norm = float(self.col_E[idx]) - 1.0
                ### logscale: log1p(x) is log(1 + x) bc this is likely skewed
                packs_norm = (np.log1p(self.col_C[idx]) - self.stats['packs_log_mean']) / self.stats['packs_log_std']
                ### z-score normalization
                age_norm = (self.col_A[idx] - self.stats['age_mean']) / self.stats['age_std']
                emph_norm = (self.col_D[idx] - self.stats['emph_mean']) / self.stats['emph_std']
                bmi_norm = (self.col_F[idx] - self.stats['bmi_mean']) / self.stats['bmi_std']
            
                if self.vars_add0 == 1:
                    meta = torch.tensor([age_norm], dtype=torch.float32)
                elif self.vars_add0 == 2:
                    meta = torch.tensor([age_norm, gender_norm], dtype=torch.float32)
                elif self.vars_add0 == 3:
                    meta = torch.tensor([age_norm, gender_norm, race_norm], dtype=torch.float32)
                elif self.vars_add0 == 4:
                    meta = torch.tensor([age_norm, gender_norm, race_norm, emph_norm], dtype=torch.float32)
                elif self.vars_add0 == 5:
                    meta = torch.tensor([age_norm, gender_norm, race_norm, emph_norm, bmi_norm], dtype=torch.float32)
                elif self.vars_add0 == 6:
                    meta = torch.tensor([age_norm, gender_norm, race_norm, emph_norm, bmi_norm, packs_norm], dtype=torch.float32)
                elif self.vars_add0 == 7:
                    meta = torch.tensor([age_norm, bmi_norm], dtype=torch.float32)
                elif self.vars_add0 == 8:
                    meta = torch.tensor([age_norm, emph_norm, bmi_norm], dtype=torch.float32)
                elif self.vars_add0 == 9:
                    meta = torch.tensor([age_norm, emph_norm], dtype=torch.float32)
                elif self.vars_add0 == 10:
                    meta = torch.tensor([emph_norm], dtype=torch.float32)
                elif self.vars_add0 == 11:
                    meta = torch.tensor([emph_norm, bmi_norm], dtype=torch.float32)
                elif self.vars_add0 == 12:
                    meta = torch.tensor([bmi_norm], dtype=torch.float32)
                elif self.vars_add0 == 13:
                    meta = torch.tensor([gender_norm], dtype=torch.float32)
                elif self.vars_add0 == 14:
                    meta = torch.tensor([race_norm], dtype=torch.float32)
                elif self.vars_add0 == 15:
                    meta = torch.tensor([packs_norm], dtype=torch.float32)            
                else:
                    raise ValueError("metadata add value not allowed, please choose between 1-15")

                return feat, meta, labels
        
            else:

                return feat, labels

    vars_add0 = vars_add
    if vars_add0 in [7,9,11]:
        vars_add = 2
    elif vars_add0 in [8]:
        vars_add = 3
    elif vars_add0 in [10,12,13,14,15]:
        vars_add = 1
    else:
        vars_add = vars_add0

    ### dim adding for age and/or gender and/or pack-years
    feat_dim = 1024
    vec_dim = feat_dim+vars_add
    original_vec_dim = feat_dim

    print(f"Vector dimension of features + metadata = {vec_dim}")

    # ---------------- MODEL ----------------
    from Utils.models.transformer_LFT import transformer_LFT
    ## available params: num_vectors, vec_dim, num_classes, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout

    task_labels = np.load(valid_labels_path)
    task_labels = task_labels.squeeze() + added

    mask = ~np.isnan(task_labels)
    task_labels = task_labels[mask]

    unique_values, counts = np.unique(task_labels, return_counts=True)
    print(f"Total Unique Values: {len(unique_values)}")
    for val, count in zip(unique_values, counts):
        print(f"Label {round(val)}: {count} occurrences")

    num_classes = int(len(unique_values))
    print("Number of classes: ", num_classes)

    task_labels = task_labels.astype(np.int64)
    counts = np.bincount(task_labels, minlength=num_classes)
    balanced_priors = torch.tensor(counts/counts.sum(), dtype=torch.float32, device=device)

    print("Priors: ",balanced_priors)
    
    dim = 512
    depth = 12
    heads = 8
    LFT_config = {
        'num_vectors': num_slices,
        'vec_dim': vec_dim,
        'num_classes': num_classes,
        'dim': dim,
        'depth': depth,
        'heads': heads,
        'mlp_dim': feat_dim,
        'pool': "cls"}

    ##### if vars_add != 0...
    class MetaTransformerWrapper(nn.Module):
        def __init__(self,
                    LFT_config: dict,
                    wrapping_mode: str = "gatt",
                    meta_input_dim: int = 4,
                    meta_hidden_dim: int = 64,
                    meta_output_dim: int = 1024,
                    original_vec_dim: int = 1024): 
            super().__init__()

            self.meta_input_dim = meta_input_dim
            self.meta_output_dim = meta_output_dim
            self.original_vec_dim = original_vec_dim
            self.wrapping_mode = wrapping_mode

            # Needed for modular (mlp)
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_input_dim, meta_hidden_dim),
                nn.LayerNorm(meta_hidden_dim),
                nn.ReLU(),
                nn.Linear(meta_hidden_dim, original_vec_dim * (2 if wrapping_mode == "modular" else 1)))

            # Needed for gate (gated)
            if self.wrapping_mode == "gate":
                self.meta_gate = nn.Sequential(
                    nn.Linear(meta_input_dim, meta_hidden_dim),
                    nn.LayerNorm(meta_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(meta_hidden_dim, original_vec_dim),
                    nn.Sigmoid())

            # Needed for gatt (gated attention)
            elif self.wrapping_mode == "gatt":
                self.query_proj = nn.Linear(meta_input_dim, original_vec_dim)
                self.key_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.attn_gate = nn.Sequential(
                    nn.Linear(original_vec_dim + meta_input_dim, original_vec_dim),
                    nn.Sigmoid())
                self.layer_norm = nn.LayerNorm(original_vec_dim)
            
            # Needed for cross (cross-attention)
            elif self.wrapping_mode == "cross":
                self.query_proj = nn.Linear(meta_input_dim, original_vec_dim)
                self.key_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.value_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.out_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.layer_norm = nn.LayerNorm(original_vec_dim)

            elif self.wrapping_mode == "dot":
                self.proj = nn.Linear(vars_add, feat_dim)

            elif self.wrapping_mode == "multi":
                self.param_weights = nn.Parameter(torch.ones(meta_input_dim))

            if self.wrapping_mode == "concat":
                new_vec_dim = vec_dim
            else:
                new_vec_dim = original_vec_dim
            
            LFT_config['vec_dim'] = new_vec_dim
            self.transformer_model = transformer_LFT(**LFT_config)

            print(f"Wrapper created. Mode: {self.wrapping_mode} | Input Dim: {new_vec_dim}")

        def forward(self, feat_embedding: torch.Tensor, meta_data: torch.Tensor):
            # Should be B=Batch, S=Slices, D=Dimension
            B, S, D = feat_embedding.shape
            fused_tensor = None

            if self.wrapping_mode == "modular": # Feat. modulation
                meta_vector = self.meta_mlp(meta_data)
                gamma, beta = meta_vector.chunk(2, dim=-1)
                gamma = gamma[:, None, :]
                beta  = beta[:, None, :]
                fused_tensor = (gamma * feat_embedding) + beta

            elif self.wrapping_mode == "concat": # Concatenate meta vars
                meta_repeated = meta_data.unsqueeze(1).repeat(1, S, 1)
                fused_tensor = torch.cat([feat_embedding, meta_repeated], dim=-1)

            elif self.wrapping_mode == "gate": # Gating by element
                gate = self.meta_gate(meta_data).unsqueeze(1) # (B, 1, D)
                fused_tensor = feat_embedding * gate

            elif self.wrapping_mode == "gatt": # Gated Feature Attention
                Q = self.query_proj(meta_data).unsqueeze(1)   # (B, 1, D)
                K = self.key_proj(feat_embedding)             # (B, S, D)
                attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                weighted_feat = feat_embedding * attn_probs.transpose(-1, -2)
                gate_input = torch.cat([Q.squeeze(1), meta_data], dim=-1)
                gate = self.attn_gate(gate_input).unsqueeze(1) # (B, 1, D)
                fused_tensor = self.layer_norm(feat_embedding + (weighted_feat * gate))

            elif self.wrapping_mode == "along": #Add the value along feature dimension
                added_val = torch.sum(meta_data)
                fused_tensor = feat_embedding*added_val

            elif self.wrapping_mode == "dot": #Dot-multiply the value by features
                meta_repeated = meta_data.unsqueeze(1).repeat(1, S, 1) # (B, S, vars_add)
                meta_projected = self.proj(meta_repeated) # (B, S, D)
                fused_tensor = torch.mul(feat_embedding, meta_projected)         
            
            elif self.wrapping_mode == "cross":
                Q = self.query_proj(meta_data).unsqueeze(1) 
                K = self.key_proj(feat_embedding)
                V = self.value_proj(feat_embedding)
                attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                context = torch.matmul(attn_weights, V)
                fused_tensor = self.layer_norm(feat_embedding + self.out_proj(context))

            elif self.wrapping_mode == "multi":
                weighted_meta = meta_data * self.param_weights
                combined_scale = torch.sum(weighted_meta, dim=-1, keepdim=True).unsqueeze(1)
                fused_tensor = feat_embedding * (1 + combined_scale)

            else:
                raise ValueError("Only accepted modes: modular, gate, gatt, cross, along, dot or concat")

            output = self.transformer_model(fused_tensor)
            
            return output

    if vars_add == 0:
        model = transformer_LFT(
            num_vectors=num_slices,
            vec_dim=vec_dim,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=feat_dim,
            pool="cls").to(device)
    else:
        model = MetaTransformerWrapper(
            LFT_config=LFT_config,
            wrapping_mode=wrapping_mode,
            meta_input_dim=vars_add,
            meta_output_dim=vec_dim,
            original_vec_dim=original_vec_dim).to(device)

    checkpoint_path = os.path.join(models_path, f"best_LFT_{vars_add0}_{wrapping_mode}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_stats = checkpoint.get("train_stats")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
            
    dataset = FeatureDataset(valid_features_path, valid_labels_path, csv_path, vars_add0, stats=train_stats)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ----------- ALLOCATION -----------
    n_samples = len(dataset)

    raw = np.zeros(n_samples)
    real = np.zeros(n_samples)
    pred = np.zeros(n_samples)

    probabs = np.zeros((n_samples, num_classes))
    probabs_raw = np.zeros((n_samples, num_classes))
    out_idx = []

    # ----------- INFERENCE LOOP -----------
    with torch.no_grad():
        for ix, complete_data in enumerate(tqdm(loader, desc="Testing patients")):
            if vars_add == 0:
                features, labels = complete_data
                if torch.isnan(labels).any():
                    out_idx.append(ix)
                    continue # Skip if whole batch was NaN
                
                labels = labels.squeeze() + added
                labels = labels.to(device).long()
                input_features = features.squeeze(2).to(device)
                outputs = model(input_features)

            else:
                features, meta_data, labels = complete_data
                if torch.isnan(labels).any():
                    out_idx.append(ix)
                    continue  # Skip if whole batch was NaN

                labels = labels.squeeze() + added
                labels = labels.to(device).long()
                input_features = features.squeeze(2).to(device)
                meta_data = meta_data.to(device)
                outputs = model(input_features,meta_data)

            logits = outputs.to(device)
            probs = torch.softmax(logits, dim=-1)

            weighted_logits = logits+torch.log(balanced_priors.unsqueeze(0))
            weighted_probs  = torch.softmax(weighted_logits, dim=-1)

            # weighted_probs = probs * task_class_weights.to(device)
            # weighted_probs = weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)
            
            pred_idx = torch.argmax(weighted_probs, dim=1).item()
            raw_idx = torch.argmax(probs, dim=1).item()

            real[ix] = labels.item() - added
            pred[ix] = pred_idx - added
            raw[ix] = raw_idx - added
            probabs[ix] = weighted_probs.cpu().numpy().flatten()
            probabs_raw[ix] = probs.cpu().numpy().flatten()
            
    print('out_idx:',out_idx)
    real = np.delete(real, out_idx, axis=0)
    pred = np.delete(pred, out_idx, axis=0)
    raw = np.delete(raw, out_idx, axis=0)
    probabs = np.delete(probabs, out_idx, axis=0)
    probabs_raw = np.delete(probabs_raw, out_idx, axis=0)    
    print(f"\nValues in real: {np.unique(real.astype(int))}")
    print(f"Values in raw: {np.unique(raw.astype(int))}")
    print(f"Values in predicted: {np.unique(pred.astype(int))}")
    n_samples_array = np.arange(n_samples)
    n_samples_clean = np.delete(n_samples_array,out_idx,axis=0)
    final_balanced_acc = accuracy_score(real, pred)*100
    raw_balanced_acc = balanced_accuracy_score(real, raw)*100
    print(f"\nAccuracy (Weighted): {final_balanced_acc:.2f}%")
    print(f"Raw Accuracy (Raw): {raw_balanced_acc:.2f}%")

    # ----------- SAVE RESULTS -----------
    
    base_csv = os.path.join(save_path, f"model_LFT_{vars_add0}_{wrapping_mode}.csv")
    probs_csv = os.path.join(save_path, f"probs_LFT_{vars_add0}_{wrapping_mode}.csv")
    raw_probs_csv =  os.path.join(save_path, f"rawprobs_LFT_{vars_add0}_{wrapping_mode}.csv")

    df = pd.DataFrame({
        "data": n_samples_clean,
        "label": real.astype(int),
        "raw": raw.astype(int),
        "predicted": pred.astype(int)})
    
    df.to_csv(base_csv, index=False)

    prob_df = pd.DataFrame(probabs, columns=[f"probs_class{i - added}" for i in range(num_classes)])
    prob_df.insert(0, "data", n_samples_clean)
    prob_df.insert(1, "label", real.astype(int))
    prob_df.to_csv(probs_csv, index=False)

    raw_prob_df = pd.DataFrame(probabs_raw, columns=[f"probs_class{i - added}" for i in range(num_classes)])
    raw_prob_df.insert(0, "data", n_samples_clean)
    raw_prob_df.insert(1, "label", real.astype(int))
    raw_prob_df.to_csv(raw_probs_csv, index=False)

    print(f"\nFinished predicting.")
    print(f"  Saved summary: {base_csv}")
    print(f"  Saved probabilities: {probs_csv}")
    print(f"  Saved raw probabilities: {raw_probs_csv}")

    return