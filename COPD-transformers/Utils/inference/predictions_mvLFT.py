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
def Predictor_mvLFT(main_path,
                num_slices,
                cuda_id,
                how,
                how_view,
                project_name,
                output_type,
                input_type,
                vars_add,
                wrapping_mode):

    # ----------- PATHS & DEVICE -----------
    cuda_set = f"cuda:{cuda_id}"
    save_path = os.path.join(main_path, f"{project_name}-results/models")
    models_path = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-mvLFT/")
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(cuda_set if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    if num_slices == 9:
        fusion_type = "C"
    elif num_slices == 20:
        fusion_type = "W"
    else:
        raise ValueError("num_slices must be 9 or 20.")
    
    added = [0,-1,1]

    # ----------- DATASET -----------
    features_dir = os.path.join(main_path, f"{project_name}-features")

    if output_type == "EMPH":
        diagnosis = "emph_cat_P1"
        model_name0 = "copd"
        IT = 0
    elif output_type == "COPD":
        diagnosis = "finalgold_visit_P1"
        model_name0 = "copd"
        IT = 2
    elif output_type == "TRAJ":
        diagnosis = "traj"
        model_name0 = "traj"
        IT = 1
    else:
        raise ValueError(f"Unsupported model: {output_type}")
    
    csv_path = os.path.join(main_path, f"{project_name}-files",f"df_val_{model_name0}.csv")

    valid_features_path_ax = os.path.join(f"{features_dir}-{num_slices}", f"{how}_test_features_{input_type}.npy")
    valid_features_path_cor = os.path.join(f"{features_dir}-{num_slices}-cor", f"{how}_test_features_{input_type}.npy")
    valid_labels_path = os.path.join(f"{features_dir}-{num_slices}", f"{how}_test_labels_{input_type}.npy")

    # ----------- MODEL LOADING -----------
    class FeatureDataset(Dataset):
        def __init__(self, features_path_ax, features_path_cor, labels_path, csv_path, vars_add0, stats=None):
            self.features_ax = np.load(features_path_ax)      # (B, slices, aug, dim)
            self.features_cor = np.load(features_path_cor)    # (B, slices, aug, dim)
            self.labels = np.load(labels_path)
            B = self.features_ax.shape[0]
            self.vars_add0 = vars_add0
            
            df = pd.read_csv(csv_path)
            if "sid" not in df.columns:
                print("WARNING: Column 0 is not named 'sid'. Using column 0 anyway.")
                df.rename(columns={df.columns[0]: "sid"}, inplace=True)

            if "age_visit_P1" not in df.columns or "gender_P1" not in df.columns or "ATS_PackYears_P1" not in df.columns or "pctEmph_Thirona_P1" not in df.columns:
                raise ValueError("CSV must contain the columns age_visit_P1, ATS_PackYears_P1, pctEmph_Thirona_P1 and gender_P1.")

            self.col_A = df["age_visit_P1"].values.astype(float)
            self.col_B = df["gender_P1"].values.astype(float)
            self.col_C = df["ATS_PackYears_P1"].values.astype(float)
            self.col_D = df["pctEmph_Thirona_P1"].values.astype(float)
            self.col_E = df["race_P1"].values.astype(float)
            self.col_F = df["BMI_P1"].values.astype(float)

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

            self.num_patients = B
            self.num_slices = self.features_ax.shape[1]
            self.num_augmentations = self.features_ax.shape[2]

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

        def __len__(self):
            return self.num_patients

        def __getitem__(self, idx):

            aug_idx = np.random.randint(self.features_ax.shape[2], size=self.features_ax.shape[1])
        
            feat_ax = self.features_ax[idx, np.arange(self.features_ax.shape[1]), aug_idx, :]
            feat_cor = self.features_cor[idx, np.arange(self.features_cor.shape[1]), aug_idx, :]
            
            feat_ax = torch.from_numpy(feat_ax).float() # (S, D)
            feat_cor = torch.from_numpy(feat_cor).float() # (S, D)
            
            labels = self.labels[idx]

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
            elif self.vars_add0 == 16:
                meta = torch.tensor([age_norm, bmi_norm/(gender_norm+1.0),emph_norm], dtype=torch.float32)            
            elif self.vars_add0 == 17:
                meta = torch.tensor([age_norm/bmi_norm+emph_norm], dtype=torch.float32)            
            elif self.vars_add0 == 18:
                meta = torch.tensor([packs_norm/emph_norm,age_norm/bmi_norm], dtype=torch.float32)            
            elif self.vars_add0 == 19:
                meta = torch.tensor([race_norm,gender_norm], dtype=torch.float32)
            else:
                meta = None

            if meta is not None:
                return feat_ax, feat_cor, meta, labels
            else:
                return feat_ax, feat_cor, labels

    ### dim adding for age and/or gender and/or pack-years...    
    
    feat_dim = 1024
    vars_add0 = vars_add

    if vars_add0 in [7,9,11,18,19]: vars_add = 2
    elif vars_add0 in [8,16]: vars_add = 3
    elif vars_add0 in [10,12,13,14,15,17]: vars_add = 1

    if len(input_type) > 1 and how == "concat":
        original_vec_dim = feat_dim * len(input_type)
    else:
        original_vec_dim = feat_dim

    vec_dim = (original_vec_dim + vars_add) if wrapping_mode == "concat" else original_vec_dim

    print(f"Vector dimension of features + metadata = {vec_dim}")

    # ---------------- MODEL ----------------

    task_labels = np.load(valid_labels_path)
    labels = task_labels[:,IT] + added[IT]
    labels[labels == -1] = 0
    unique_values, counts = np.unique(labels, return_counts=True)
    print(f"Total Unique Values: {len(unique_values)}")
    for val, count in zip(unique_values, counts):
        print(f"Label {(val)}: {count} occurrences")

    num_classes = int(len(unique_values))

    # ---------------- MODEL ----------------
    from Utils.models.transformer_LFT import transformer_LFT
    ## available params: num_vectors, vec_dim, num_classes, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout

    dim = 512
    depth = 12
    heads = 8
    mvLFT_config = {
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
                    mvLFT_config: dict,
                    wrapping_mode: str = "gatt",
                    how_view: str = "cross",
                    meta_input_dim: int = 4,
                    meta_hidden_dim: int = 64,
                    original_vec_dim: int = 1024): 
            super().__init__()

            self.meta_input_dim = meta_input_dim
            self.original_vec_dim = original_vec_dim
            self.wrapping_mode = wrapping_mode
            self.how_view = how_view

            # --- PHASE 1: VIEW FUSION ---

            # PCA/Linear projection expects the full 2048 concatenated input
            if self.how_view == "pca":
                self.view_proj = nn.Linear(original_vec_dim * 2, original_vec_dim)
            
            # Gated mixing expects full 2048 input to produce 2 weights
            elif self.how_view == "gated":
                self.view_gate_mlp = nn.Sequential(
                    nn.Linear(original_vec_dim * 2, 2),
                    nn.Softmax(dim=-1))
            
            # Learned Query Attention
            elif self.how_view == "att":
                self.view_query = nn.Parameter(torch.randn(1, 1, original_vec_dim))
            
            # Gated Attention (GATT) between views
            elif self.how_view == "gatt":
                self.view_q_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.view_k_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.view_attn_gate = nn.Sequential(
                    nn.Linear(original_vec_dim * 2, original_vec_dim),
                    nn.Sigmoid())
                self.view_norm = nn.LayerNorm(original_vec_dim)
            
            # Standard Cross-Attention between views
            elif self.how_view == "cross":
                self.view_q_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.view_k_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.view_v_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.view_out_proj = nn.Linear(original_vec_dim, original_vec_dim)
                self.view_norm = nn.LayerNorm(original_vec_dim)

            self.current_fused_dim = original_vec_dim * 2 if how_view == "concat" else original_vec_dim

            # --- PHASE 2: METADATA WRAPPING ---

            if self.meta_input_dim > 0:
                if self.wrapping_mode == "modular":
                    self.meta_mlp = nn.Sequential(
                        nn.Linear(meta_input_dim, meta_hidden_dim),
                        nn.LayerNorm(meta_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(meta_hidden_dim, self.current_fused_dim * 2))
                
                elif self.wrapping_mode == "gate":
                    self.meta_gate = nn.Sequential(
                        nn.Linear(meta_input_dim, meta_hidden_dim),
                        nn.LayerNorm(meta_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(meta_hidden_dim, self.current_fused_dim),
                        nn.Sigmoid())
                
                elif self.wrapping_mode == "gatt":
                    self.query_proj = nn.Linear(meta_input_dim, self.current_fused_dim)
                    self.key_proj = nn.Linear(self.current_fused_dim, self.current_fused_dim)
                    self.attn_gate = nn.Sequential(
                        nn.Linear(self.current_fused_dim * 2, self.current_fused_dim),
                        nn.Sigmoid())
                    self.layer_norm = nn.LayerNorm(self.current_fused_dim)
                
                elif self.wrapping_mode == "cross":
                    self.query_proj = nn.Linear(meta_input_dim, self.current_fused_dim)
                    self.key_proj = nn.Linear(self.current_fused_dim, self.current_fused_dim)
                    self.value_proj = nn.Linear(self.current_fused_dim, self.current_fused_dim)
                    self.out_proj = nn.Linear(self.current_fused_dim, self.current_fused_dim)
                    self.layer_norm = nn.LayerNorm(self.current_fused_dim)
                
                elif self.wrapping_mode == "dot":
                    self.proj = nn.Linear(meta_input_dim, self.current_fused_dim)
                
                elif self.wrapping_mode == "multi":
                    self.param_weights = nn.Parameter(torch.ones(meta_input_dim))

            # --- TRANSFORMER INIT ---
            transformer_in_dim = self.current_fused_dim + meta_input_dim if wrapping_mode == "concat" else self.current_fused_dim
            mvLFT_config['vec_dim'] = transformer_in_dim
            
            self.transformer_model = transformer_LFT(**mvLFT_config)

        def forward(self, feat_embedding: torch.Tensor, meta_data: torch.Tensor = None):
            B, S, Total_D = feat_embedding.shape
            
            # --- PHASE 1: VIEW FUSION ---
            axial, coronal = torch.chunk(feat_embedding, 2, dim=-1)
            
            if self.how_view == "concat":
                v_fused = feat_embedding
            
            elif self.how_view == "cross":
                v_q, v_k, v_v = self.view_q_proj(axial), self.view_k_proj(coronal), self.view_v_proj(coronal)
                attn = torch.softmax(torch.matmul(v_q, v_k.transpose(-1, -2)) / (self.original_vec_dim**0.5), dim=-1)
                v_fused = self.view_norm(axial + self.view_out_proj(torch.matmul(attn, v_v)))
            
            elif self.how_view == "gatt":
                v_q, v_k = self.view_q_proj(coronal), self.view_k_proj(axial)
                attn = torch.softmax(torch.matmul(v_q, v_k.transpose(-1, -2)) / (self.original_vec_dim**0.5), dim=-1)
                interacted = torch.matmul(attn, axial)
                gate = self.view_attn_gate(torch.cat([v_q, interacted], dim=-1))
                v_fused = self.view_norm(coronal + (interacted * gate))
            
            elif self.how_view == "gated":
                w = self.view_gate_mlp(feat_embedding) # [B, S, 2]
                v_fused = w[:, :, 0:1] * axial + w[:, :, 1:2] * coronal
            
            elif self.how_view == "att":
                stacked = torch.stack([axial, coronal], dim=2) # [B, S, 2, 1024]
                attn = torch.softmax(torch.matmul(self.view_query, stacked.transpose(-1, -2)) / (self.original_vec_dim**0.5), dim=-1)
                v_fused = torch.matmul(attn, stacked).squeeze(2)
            
            elif self.how_view == "mean":
                v_fused = (axial + coronal) / 2
            
            elif self.how_view == "max":
                v_fused = torch.max(axial, coronal)
            
            elif self.how_view == "min":
                v_fused = torch.min(axial, coronal)
            
            elif self.how_view == "std":
                v_fused = torch.std(torch.stack([axial, coronal], dim=2), dim=2)
            
            elif self.how_view == "var":
                v_fused = torch.var(torch.stack([axial, coronal], dim=2), dim=2)
            
            elif self.how_view == "pca":
                v_fused = self.view_proj(feat_embedding)
            
            else:
                v_fused = axial

            # --- PHASE 2: METADATA WRAPPING ---
            if self.meta_input_dim == 0 or meta_data is None:
                return self.transformer_model(v_fused)

            if self.wrapping_mode == "modular":
                gamma, beta = self.meta_mlp(meta_data).chunk(2, dim=-1)
                v_fused = (gamma.unsqueeze(1) * v_fused) + beta.unsqueeze(1)
            
            elif self.wrapping_mode == "concat":
                v_fused = torch.cat([v_fused, meta_data.unsqueeze(1).repeat(1, S, 1)], dim=-1)
            
            elif self.wrapping_mode == "gate":
                v_fused = v_fused * self.meta_gate(meta_data).unsqueeze(1)
            
            elif self.wrapping_mode == "gatt":
                # meta_data: [B, meta_dim] -> Q: [B, 1, D]
                Q = self.query_proj(meta_data).unsqueeze(1) 
                # v_fused: [B, S, D] -> K: [B, S, D], V: [B, S, D]
                K = self.key_proj(v_fused)
                V = v_fused
                attn = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / (self.current_fused_dim**0.5), dim=-1)
                weighted = torch.matmul(attn, V) 
                # gate: [B, 1, D]
                gate_input = torch.cat([Q, weighted], dim=-1)
                gate = self.attn_gate(gate_input)
                v_fused = self.layer_norm(v_fused + (weighted * gate))
            
            elif self.wrapping_mode == "cross":
                Q = self.query_proj(meta_data).unsqueeze(1)
                K, V = self.key_proj(v_fused), self.value_proj(v_fused)
                attn = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / (self.current_fused_dim**0.5), dim=-1)
                v_fused = self.layer_norm(v_fused + self.out_proj(torch.matmul(attn, V)))
            
            elif self.wrapping_mode == "dot":
                v_fused = v_fused * self.proj(meta_data).unsqueeze(1)
            
            elif self.wrapping_mode == "multi":
                v_fused = v_fused * (1 + torch.sum(meta_data * self.param_weights, dim=-1, keepdim=True).unsqueeze(1))

            return self.transformer_model(v_fused)

    # --- FACTORY INSTANTIATION ---

    model = MetaTransformerWrapper(
        mvLFT_config=mvLFT_config,     # Dict containing transformer params
        wrapping_mode=wrapping_mode,   # 'gatt', 'modular', 'gate', etc.
        how_view=how_view,             # 'cross', 'gatt', 'mean', etc.
        meta_input_dim=vars_add,       # Number of clinical variables (e.g., 4)
        original_vec_dim=original_vec_dim).to(device) # Base dimension per view (e.g., 1024)

    checkpoint_path = os.path.join(models_path, f"best_mvLFT_{output_type}_{fusion_type}-{how}_{how_view}_{input_type}_{vars_add0}_{wrapping_mode}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_stats = checkpoint.get("train_stats")
    model.load_state_dict(checkpoint["model_state_dict"],strict=False)
    model.eval()
            
    dataset = FeatureDataset(valid_features_path_ax, valid_features_path_cor, valid_labels_path, csv_path, vars_add0, stats=train_stats)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    task_labels = task_labels[:,IT] + added[IT]
    task_shifted = task_labels.astype(np.int64)
    task_shifted[task_shifted == -1] = 0
    counts = np.bincount(task_shifted, minlength=num_classes)
    priors_task = torch.tensor(counts/counts.sum(), dtype=torch.float32, device=device)

    # ----------- ALLOCATION -----------
    n_samples = len(dataset)

    raw = np.zeros(n_samples)
    real = np.zeros(n_samples)
    pred = np.zeros(n_samples)

    probabs = np.zeros((n_samples, num_classes))
    probabs_raw = np.zeros((n_samples, num_classes))

    # ----------- INFERENCE LOOP -----------
    with torch.no_grad():
        for ix, complete_data in enumerate(tqdm(loader, desc="Testing patients")):
            if vars_add == 0:
                feat_ax, feat_cor, labels = complete_data
                meta_data = None
            else:
                feat_ax, feat_cor, meta_data, labels = complete_data
                meta_data = meta_data.to(device)

            labels = labels[:,IT].to(device).long() + added[IT]
            labels[labels == -1] = 0            
            
            # Fuse views into one tensor for the Wrapper [B, S, 2048]
            input_features = torch.cat([feat_ax, feat_cor], dim=-1).to(device)

            outputs = model(input_features, meta_data)

            logits = outputs.to(device)
            
            probs = torch.softmax(logits, dim=-1)

            weighted_logits = logits + torch.log(priors_task.unsqueeze(0))
            weighted_probs  = torch.softmax(weighted_logits, dim=-1)

            # weighted_probs = probs * task_class_weights.to(device)
            # weighted_probs = weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)
            
            pred_idx = torch.argmax(weighted_probs, dim=1).item()
            raw_idx = torch.argmax(probs, dim=1).item()

            real[ix] = labels.item()
            pred[ix] = pred_idx
            raw[ix] = raw_idx
            probabs[ix] = weighted_probs.cpu().numpy().flatten()
            probabs_raw[ix] = probs.cpu().numpy().flatten()
            
    print(f"\nValues in real: {np.unique(real.astype(int))}")
    print(f"Values in raw: {np.unique(raw.astype(int))}")
    print(f"Values in predicted: {np.unique(pred.astype(int))}")

    final_balanced_acc = accuracy_score(real, pred)*100
    raw_balanced_acc = balanced_accuracy_score(real, raw)*100
    print(f"\nAccuracy (Weighted): {final_balanced_acc:.2f}%")
    print(f"Balanced Accuracy (Raw): {raw_balanced_acc:.2f}%")

    # ----------- SAVE RESULTS -----------
    base_csv = os.path.join(save_path, f"model_mvLFT_{output_type}_{fusion_type}-{how}_{how_view}_{input_type}_{vars_add0}_{wrapping_mode}.csv")
    probs_csv = os.path.join(save_path, f"probs_mvLFT_{output_type}_{fusion_type}-{how}_{how_view}_{input_type}_{vars_add0}_{wrapping_mode}.csv")
    raw_probs_csv =  os.path.join(save_path, f"rawprobs_mvLFT_{output_type}_{fusion_type}-{how}_{how_view}_{input_type}_{vars_add0}_{wrapping_mode}.csv")
    df = pd.DataFrame({
        "data": np.arange(n_samples),
        "label": real.astype(int),
        "raw": raw.astype(int),
        "predicted": pred.astype(int)})
    df.to_csv(base_csv, index=False)

    prob_df = pd.DataFrame(probabs, columns=[f"probs_class{i - added[IT]}" for i in range(num_classes)])
    prob_df.insert(0, "data", np.arange(n_samples))
    prob_df.insert(1, "label", real.astype(int))
    prob_df.to_csv(probs_csv, index=False)

    raw_prob_df = pd.DataFrame(probabs_raw, columns=[f"probs_class{i - added[IT]}" for i in range(num_classes)])
    raw_prob_df.insert(0, "data", np.arange(n_samples))
    raw_prob_df.insert(1, "label", real.astype(int))
    raw_prob_df.to_csv(raw_probs_csv, index=False)

    print(f"\nFinished predicting for {output_type}.")
    print(f"  Saved summary: {base_csv}")
    print(f"  Saved probabilities: {probs_csv}")
    print(f"  Saved raw probabilities: {raw_probs_csv}")

    return