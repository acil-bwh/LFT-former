# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2026 ··············
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
from torch.utils.data import Dataset

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

# ---------------------- Dataset Class ----------------------
class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path, csv_path, vars_add0, stats=None):
        self.features = np.load(features_path)      # (B, slices, aug, dim)
        self.labels = np.load(labels_path)          # (B, num_tasks)
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
        return len(self.features)

    def __getitem__(self, idx):
        # Deterministic slice selection for single prediction (or random)
        num_slices = self.features.shape[1]
        num_augmentations = self.features.shape[2]
        
        # Use first augmentation for stability in single prediction, or random
        aug_idx = np.zeros(num_slices, dtype=int) 
        feat = self.features[idx, np.arange(num_slices), aug_idx, :]
        feat = torch.from_numpy(feat[:, None, :]).float()
        
        labels = self.labels[idx]

        # Normalization logic
        gender_norm = float(self.col_B[idx]) - 1.0
        race_norm = float(self.col_E[idx]) - 1.0
        packs_norm = (np.log1p(self.col_C[idx]) - self.stats['packs_log_mean']) / self.stats['packs_log_std']
        age_norm = (self.col_A[idx] - self.stats['age_mean']) / self.stats['age_std']
        emph_norm = (self.col_D[idx] - self.stats['emph_mean']) / self.stats['emph_std']
        bmi_norm = (self.col_F[idx] - self.stats['bmi_mean']) / self.stats['bmi_std']

        # Meta vector construction
        meta_map = {
            1: [age_norm], 2: [age_norm, gender_norm], 3: [age_norm, gender_norm, race_norm],
            4: [age_norm, gender_norm, race_norm, emph_norm], 
            5: [age_norm, gender_norm, race_norm, emph_norm, bmi_norm],
            6: [age_norm, gender_norm, race_norm, emph_norm, bmi_norm, packs_norm],
            7: [age_norm, bmi_norm], 8: [age_norm, emph_norm, bmi_norm], 9: [age_norm, emph_norm],
            10: [emph_norm], 11: [emph_norm, bmi_norm], 12: [bmi_norm], 13: [gender_norm],
            14: [race_norm], 15: [packs_norm]}
        
        meta = torch.tensor(meta_map.get(self.vars_add0, []), dtype=torch.float32) if self.vars_add0 > 0 else None
        return (feat, meta, labels) if meta is not None else (feat, labels)

# ---------------------- Prediction Function ----------------------
def Probabilities_LFT(main_path, cuda_id, id, project_name, vars_add0, wrapping_mode, num_slices=20):
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    
    # Task Setup
    added_val = -1
    
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

    # Paths
    feat_p = os.path.join(main_path, f"{project_name}-features", f"test_features.npy")
    lab_p = os.path.join(main_path, f"{project_name}-features", f"test_labels.npy")
    csv_p = os.path.join(main_path, f"{project_name}-files", f"df_val_traj.csv")
    ckpt_p = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-LFT/", f"best_LFT_{vars_add0}_{wrapping_mode}.pt")

    # 1. Load Checkpoint (Fixing PyTorch 2.6+ issue)
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    checkpoint = torch.load(ckpt_p, map_location=device, weights_only=False)
    
    # 2. Setup Model
    ds = FeatureDataset(feat_p, lab_p, csv_p, vars_add0, stats=checkpoint.get("train_stats"))
    num_classes = int(len(np.unique(ds.labels)))

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

    # ---------------------- Model Wrapper ----------------------
    from Utils.models.transformer_LFT import transformer_LFT
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
        
    # 3. Fix State Dict Keys
    sd = checkpoint["model_state_dict"]
    fixed_sd = { (f"transformer_model.{k}" if not k.startswith("transformer_model.") and "proj" not in k else k): v for k, v in sd.items()}
    model.load_state_dict(fixed_sd, strict=False)
    model.eval()

    # 4. Inference
    try:
        idx = ds.sids.index(str(id))
    except ValueError:
        print(f"ID {id} not found."); return

    data = ds[idx]
    with torch.no_grad():
        feat = data[0].unsqueeze(0).squeeze(2).to(device)
        meta = data[1].unsqueeze(0).to(device) if vars_add0 > 0 else None
        label_raw = data[2] if vars_add0 > 0 else data[1]
        logits = model(feat, meta)
        
        # Bayesian Priors
        shifted = ds.labels.astype(int) + added_val
        priors = torch.tensor(np.bincount(shifted, minlength=num_classes)/len(shifted), device=device)
        probs_raw = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        probs_weighted = torch.softmax(logits + torch.log(priors + 1e-9), dim=-1).cpu().numpy().flatten()
        pred = np.argmax(probs_weighted)
        true_idx = int(label_raw) + added_val

    print(f"\n{'='*50}")
    print(f"RESULTS FOR PATIENT: {id}")
    print(f"{'='*50}")
    print(f"{'Label':<12} | {'Raw Prob':<10} | {'Weighted Prob'}")
    print(f"{'-'*45}")
    
    for i in range(num_classes):
        # Calculate real-world category name
        cat_name = f"traj_{i - added_val}"
        
        # Add Markers
        markers = []
        if i == pred: markers.append("[PREDICTED]")
        if i == true_idx: markers.append("[TRUE LABEL]")
        marker_str = " ".join(markers)
        
        print(f"{cat_name:<12} | {probs_raw[i]:.4f}     | {probs_weighted[i]:.4f}  {marker_str}")

    print(f"{'-'*45}")
    print(f"Final Prediction: traj_{pred - added_val}")
    print(f"Actual Value:     traj_{true_idx - added_val}")
    print(f"Result:           {'CORRECT' if pred == true_idx else 'INCORRECT'}")
    print(f"{'='*50}\n")

    return probs_weighted