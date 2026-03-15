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
            df.rename(columns={df.columns[0]: "sid"}, inplace=True)

        self.sids = df["sid"].astype(str).tolist()
        self.col_A = df["age_visit_P1"].values.astype(float)
        self.col_B = df["gender_P1"].values.astype(float)
        self.col_C = df["ATS_PackYears_P1"].values.astype(float)
        self.col_D = df["pctEmph_Thirona_P1"].values.astype(float)
        self.col_E = df["race_P1"].values.astype(float)
        self.col_F = df["BMI_P1"].values.astype(float)

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
            14: [race_norm], 15: [packs_norm], 16: [age_norm, bmi_norm/(gender_norm+1.0), emph_norm],
            17: [age_norm/bmi_norm+emph_norm], 18: [packs_norm/emph_norm, age_norm/bmi_norm]}
        
        meta = torch.tensor(meta_map.get(self.vars_add0, []), dtype=torch.float32) if self.vars_add0 > 0 else None
        return (feat, meta, labels) if meta is not None else (feat, labels)

# ---------------------- Model Wrapper ----------------------
from Utils.models.transformer_LFT import transformer_LFT

class MetaTransformerWrapper(nn.Module):
    def __init__(self, LFT_config, wrapping_mode="gatt", meta_input_dim=1, original_vec_dim=1024):
        super().__init__()
        self.wrapping_mode = wrapping_mode
        if wrapping_mode == "gatt":
            self.query_proj = nn.Linear(meta_input_dim, original_vec_dim)
            self.key_proj = nn.Linear(original_vec_dim, original_vec_dim)
            self.attn_gate = nn.Sequential(nn.Linear(original_vec_dim + meta_input_dim, original_vec_dim), nn.Sigmoid())
            self.layer_norm = nn.LayerNorm(original_vec_dim)
        
        self.transformer_model = transformer_LFT(**LFT_config)

    def forward(self, feat_embedding, meta_data=None):
        # Fix: make meta_data optional
        if meta_data is None or self.wrapping_mode == "none":
            return self.transformer_model(feat_embedding)
        
        B, S, D = feat_embedding.shape
        if self.wrapping_mode == "gatt":
            Q = self.query_proj(meta_data).unsqueeze(1)
            K = self.key_proj(feat_embedding)
            attn = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / (D**0.5), dim=-1)
            weighted = feat_embedding * attn.transpose(-1, -2)
            gate = self.attn_gate(torch.cat([Q.squeeze(1), meta_data], dim=-1)).unsqueeze(1)
            fused = self.layer_norm(feat_embedding + (weighted * gate))
            return self.transformer_model(fused)
        return self.transformer_model(feat_embedding)

# ---------------------- Prediction Function ----------------------
def Probabilities_LFT(main_path, num_slices, cuda_id, id, how, project_name, output_type, input_type, vars_add0, wrapping_mode):
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    
    # Task Setup
    task_info = {"EMPH": (0, 0, "copd"), "TRAJ": (1, -1, "traj"), "COPD": (2, 1, "copd")}
    IT, added_val, model_base = task_info[output_type]
    
    actual_vars = vars_add0
    if vars_add0 in [7,9,11,18]: actual_vars = 2
    elif vars_add0 in [8,16]: actual_vars = 3
    elif vars_add0 in [10,12,14,15,17]: actual_vars = 1

    # Paths
    feat_p = os.path.join(main_path, f"{project_name}-features-{num_slices}", f"{how}_test_features_{input_type}.npy")
    lab_p = os.path.join(main_path, f"{project_name}-features-{num_slices}", f"{how}_test_labels_{input_type}_{IT+1}.npy")
    csv_p = os.path.join(main_path, f"{project_name}-files", f"df_val_{model_base}.csv")
    ckpt_p = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-LFT/", f"best_LFT_{output_type}_{'C' if num_slices==9 else 'W'}-{how}_{input_type}_{vars_add0}_{wrapping_mode}.pt")

    # 1. Load Checkpoint (Fixing PyTorch 2.6+ issue)
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    checkpoint = torch.load(ckpt_p, map_location=device, weights_only=False)
    
    # 2. Setup Model
    ds = FeatureDataset(feat_p, lab_p, csv_p, vars_add0, stats=checkpoint.get("train_stats"))
    num_classes = int(len(np.unique(ds.labels)))
    LFT_config = {'num_vectors': num_slices, 'vec_dim': 1024, 'num_classes': num_classes, 'dim': 512, 'depth': 12, 'heads': 8, 'mlp_dim': 1024, 'pool': "cls"}
    
    model = MetaTransformerWrapper(LFT_config, wrapping_mode, actual_vars, 1024).to(device)

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
        cat_name = f"{output_type}_{i - added_val}"
        
        # Add Markers
        markers = []
        if i == pred: markers.append("[PREDICTED]")
        if i == true_idx: markers.append("[TRUE LABEL]")
        marker_str = " ".join(markers)
        
        print(f"{cat_name:<12} | {probs_raw[i]:.4f}     | {probs_weighted[i]:.4f}  {marker_str}")

    print(f"{'-'*45}")
    print(f"Final Prediction: {output_type}_{pred - added_val}")
    print(f"Actual Value:     {output_type}_{true_idx - added_val}")
    print(f"Result:           {'CORRECT' if pred == true_idx else 'INCORRECT'}")
    print(f"{'='*50}\n")

    return probs_weighted