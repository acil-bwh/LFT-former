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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
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

def balanced_accuracy_torch(outputs, labels, num_classes, average="micro"):
    if outputs.ndim == 2:
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = outputs
    
    # Ensure labels are long and on the same device
    labels = labels.to(preds.device).long()
    correct = (preds == labels).float()

    if average == "micro":
        return correct.mean().item()

    # Macro: Calculate mean of accuracies per class
    acc_per_class = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any(): # .any() is safer and faster than .sum() == 0
            acc_per_class.append(correct[mask].mean())
    
    if not acc_per_class:
        return 0.0
    
    return torch.stack(acc_per_class).mean().item()

def compute_class_weights(test_labels, num_classes, device):
    test_labels = test_labels.astype(int)
    counts = np.bincount(test_labels, minlength=num_classes)
    counts = np.maximum(counts, 1) 
    freqs = counts/counts.sum()
    weights = 1.0/freqs
    weights = weights*(num_classes/weights.sum())
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ---------------- TRAINING FUNCTION ----------------

def Trainer_LFT(main_path,
               num_slices,
               cuda_id,
               project_name,
               epochs,
               batch_size,
               load_pretrained,
               how,
               load_from_checkpoint,
               lr,
               gamma,
               step,
               optzr,
               schdr,
               weight_decay,
               output_type,
               input_type,
               vars_add,
               wrapping_mode,
               axis):
    
    """
    Train a multi-slice Trainer_clinLFT model with patient-level features.

    Args:
        main_path (str): Root directory path.
        num_slices (int): Slices per patient (e.g., 9 or 20).
        cuda_id (int): GPU device ID.
        project_name (str): Project prefix.
        epochs (int): Training epochs.
        batch_size (int): Batch size.
        how (str): Type of fusion of multimodal weights, if any.
        lr (float): Learning rate.
        weight_decay (float): L2 weight decay.
        gamma (float): LR scheduler decay.
        load_pretrained (bool): Whether to initialize weights from pretrained model.
        load_from_checkpoint (bool): Whether to resume from checkpoint.
        input_type (str): E, C, T, EC, ET, CT, ETC.
        add (int): 1: age, 2: age+gender, 3: age+gender+packs
        wrapping_mode (str): concat, modular, gatt, gate
    """    
    
    print("\n================ Trainer_clinLFT Configuration ================")
    print(f" Main path: {main_path}")
    print(f" Project: {project_name}")
    print(f" Loading from: {project_name}")
    print(f" Num slices: {num_slices}")
    print(f" CUDA device: {cuda_id}")
    print(f" Epochs: {epochs}")
    print(f" Batch size: {batch_size}")
    print(f" Fusion of weights: {how}")
    print(f" Optimizer: {optzr}")
    print(f" Scheduler: {schdr}")
    print(f" Gamma: {gamma}")
    print(f" Step: {step}")
    print(f" Learning rate: {lr}")
    print(f" Weight decay: {weight_decay}")
    print(f" Which input: {input_type}")
    print(f" Which output: {output_type}")
    print(f" Wrapping mode: {wrapping_mode}")
    print(f" Pretrained model: {load_pretrained}")
    print(f" Resume checkpoint: {load_from_checkpoint}")
    print(f" Axis of 2D slices: {axis}")
    print("========================================================\n")

    # ---------------- DEVICE ----------------
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # ---------------- PATHS ----------------
    main_checkpoint_path = os.path.join(main_path, f"{project_name}-checkpoints/checkpoints-LFT")
    os.makedirs(main_checkpoint_path, exist_ok=True)
    
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
    elif output_type == "TRAP":
        diagnosis = "trap"
        model_name0 = "gas"
        IT = 3
    else:
        raise ValueError(f"Unsupported model: {output_type}")

    if num_slices == 9:
        fusion_type = "C"
    elif num_slices == 20:
        fusion_type = "W"
    else:
        raise ValueError("num_slices must be 9 or 20.")

    added = [0,-1,1,0]
 
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
                required_cols = ["age_visit_P1", "gender_P1", "ATS_PackYears_P1", "pctEmph_Thirona_P1", "race_P1", "BMI_P1"]
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"CSV missing required column: {col}")
                    
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
                elif self.vars_add0 == 16:
                    meta = torch.tensor([age_norm, bmi_norm/(gender_norm+1.0),emph_norm], dtype=torch.float32)            
                elif self.vars_add0 == 17:
                    meta = torch.tensor([age_norm/bmi_norm+emph_norm], dtype=torch.float32)            
                elif self.vars_add0 == 18:
                    meta = torch.tensor([packs_norm/emph_norm,age_norm/bmi_norm], dtype=torch.float32)            
                elif self.vars_add0 == 19:
                    meta = torch.tensor([gender_norm,race_norm], dtype=torch.float32)            
                else:
                    meta = torch.tensor([age_norm], dtype=torch.float32)

                return feat, meta, labels
        
            return feat, labels

    vars_add0 = vars_add
    if vars_add0 in [7,9,11,18,19]:
        vars_add = 2
    elif vars_add0 in [8,16]:
        vars_add = 3
    elif vars_add0 in [10,12,13,14,15,17]:
        vars_add = 1
    else:
        vars_add = vars_add0

    ### dim adding for age and/or gender and/or pack-years...
    feat_dim = 1024
    if len(input_type) != 1:
        if how == "concat":
            vec_dim = feat_dim*len(input_type)+vars_add
            original_vec_dim = feat_dim*len(input_type)
        elif how == "mix":
            vec_dim = feat_dim*(len(input_type)-1)+vars_add
            original_vec_dim = feat_dim*(len(input_type)-1)
        else:
            vec_dim = feat_dim+vars_add
            original_vec_dim = feat_dim
    else:
        vec_dim = feat_dim+vars_add
        original_vec_dim = feat_dim

    print(f"Vector dimension of features + metadata = {vec_dim}")

    csv_path = os.path.join(main_path, f"{project_name}-files",f"df_train_{model_name0}.csv")

    # ------------------ Create dataset ------------------

    if axis == "coronal":
        axis_suffix = "-cor"
    elif axis == "axial":
        axis_suffix = ""

    from torch.utils.data import random_split
    print(f"Number of embeddings used for the model: {len(input_type)}")

    if len(input_type) == 1:
        features_dir = os.path.join(main_path, f"{project_name}-embeddings/{project_name}{IT+1}-features")
        features_path = os.path.join(f"{features_dir}-{num_slices}{axis_suffix}", f"train_features.npy")
        labels_path = os.path.join(f"{features_dir}-{num_slices}{axis_suffix}", f"train_labels.npy")
    else:
        features_dir = os.path.join(main_path, f"{project_name}-embeddings/{project_name}-features")
        features_path = os.path.join(f"{features_dir}-{num_slices}{axis_suffix}", f"{how}_train_features_{input_type}.npy")
        labels_path = os.path.join(f"{features_dir}-{num_slices}{axis_suffix}", f"{how}_train_labels_{input_type}.npy")
    
    print(f"\Full dataset loading from {features_path} and {labels_path}")

    full_dataset = FeatureDataset(features_path, labels_path, csv_path, vars_add0, stats=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"\nTraining: loading data {train_size}...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"\nValidation: loading data {val_size}...")
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if vars_add != 0:
        train_stats = full_dataset.stats
        val_dataset.dataset.stats = train_stats

    labels = np.load(labels_path)
    labels = labels[:,IT].squeeze() + added[IT]
    if output_type == "COPD":
        labels[labels == -1] = 0

    unique_values, counts = np.unique(labels, return_counts=True)
    print(f"Total Unique Values: {len(unique_values)}")
    print(f"Unique values: ", unique_values)
    for val, count in zip(unique_values, counts):
        print(f"Label {round(val)}: {count} occurrences")

    num_classes = len(unique_values)
    task_class_weights = compute_class_weights(labels, num_classes, device)
    print("Weights per class: ", task_class_weights)
    print("Shape of weights: ", task_class_weights.shape)

    # ---------------- MODEL ----------------
    from Utils.models.transformer_LFT import transformer_LFT
    from Utils.models.basicViT import ViT
    ## available params: num_vectors, vec_dim, num_classes, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout

    ##### if vars_add != 0...
    class MetaTransformerWrapper(nn.Module):
        def __init__(self,
                    LFT_config: dict,
                    model_type: str = "lft", # "lft" or "vit"
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

            if model_type.lower() == "vit":
                self.transformer_model = ViT(**LFT_config)
            else:
                self.transformer_model = transformer_LFT(**LFT_config)

            print(f"Wrapper created. Model: {model_type} | Mode: {self.wrapping_mode}")

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

    model_choice = "vit" # Change to "lft" for the LFT version
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

    if vars_add == 0:
        if model_choice == "vit":
            model = ViT(**LFT_config).to(device)
        else:
            model = transformer_LFT(**LFT_config).to(device)
    else:
        model = MetaTransformerWrapper(
            LFT_config=LFT_config,
            model_type=model_choice, # Pass choice here
            wrapping_mode=wrapping_mode,
            meta_input_dim=vars_add,
            meta_output_dim=vec_dim,
            original_vec_dim=original_vec_dim).to(device)

    if optzr == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optzr == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=gamma, weight_decay=weight_decay)
    
    if schdr == "plat":
        scheduler = ReduceLROnPlateau(optimizer,
        mode='min',
        factor=gamma,
        patience=step,
        min_lr=1e-9)
    elif schdr == "step":
        scheduler = StepLR(optimizer, step_size=step, gamma=gamma)
    
    if load_pretrained == True and load_from_checkpoint == False:
        checkpoint_path = os.path.join(main_path, f"{project_name}-models/model_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add0}_{wrapping_mode}{axis_suffix}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        last_epoch = 0
        best_val_acc = 0.0
        print(f"Loaded pretrained weights from {checkpoint_path}\n")

    elif load_from_checkpoint == True and load_pretrained == False:
        checkpoint_path = os.path.join(main_checkpoint_path, f"best_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add0}_{wrapping_mode}{axis_suffix}.pt")
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
    
    best_epoch = last_epoch
    epochs_no_improve = 0
    patience = 20
    best_val_loss = float('inf')

    criterion = nn.CrossEntropyLoss(weight=task_class_weights)
    
    for epoch in range(last_epoch, last_epoch + epochs):
        model.train()
        total_loss, total_acc = 0, 0
        optimizer.zero_grad()

        for complete_data in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            if vars_add == 0:
                features, labels = complete_data
                input_features = features.squeeze(2).to(device)
                outputs = model(input_features)
            else:
                features, meta_data, labels = complete_data
                input_features = features.squeeze(2).to(device)
                meta_data = meta_data.to(device)
                outputs = model(input_features,meta_data)

            labels = labels[:,IT].squeeze()
            labels = labels.to(device).long() + added[IT]
            if output_type == "COPD":
                labels[labels == -1] = 0
            
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)

            acc = balanced_accuracy_torch(outputs,labels,num_classes,average="macro")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)

        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_acc)

        model.eval()
        val_loss, val_acc = 0, 0

        with torch.no_grad():
            for complete_data in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                if vars_add == 0:
                    features, labels = complete_data
                    input_features = features.squeeze(2).to(device)
                    outputs = model(input_features)
                else:
                    features, meta_data, labels = complete_data
                    input_features = features.squeeze(2).to(device)
                    meta_data = meta_data.to(device)
                    outputs = model(input_features,meta_data)

                labels = labels[:,IT].squeeze()
                labels = labels.to(device).long() + added[IT]
                if output_type == "COPD":
                    labels[labels == -1] = 0
                    
                outputs = outputs.to(device)

                loss = criterion(outputs, labels)
                acc = balanced_accuracy_torch(outputs,labels,num_classes,average="macro")
                    
                val_loss += loss.item()
                val_acc += acc

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_acc = val_acc / len(valid_loader)

        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_acc)

        if schdr == "plat":
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

            best_model_path = os.path.join(main_checkpoint_path, f"best_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add0}_{wrapping_mode}{axis_suffix}.pt")
            
            if vars_add !=0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_stats": train_stats,
                    "val_acc": best_val_acc}, best_model_path)
            else:
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
        
    final_model_path = os.path.join(main_path, f"{project_name}-models/model_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add0}_{wrapping_mode}{axis_suffix}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": validation_loss[-1]}, final_model_path)

    print("\nTraining complete.")
    print(f"Saved final model at: {final_model_path}")
    print(f"Best LFT using {input_type} for {output_type} model axis: {axis_suffix} adding: {vars_add0} and wrapping {wrapping_mode} saved at epoch {last_epoch+best_epoch} with val_acc: {best_val_acc:.4f}")

    return {"train_loss": training_loss,
        "val_loss": validation_loss,
        "train_acc": training_acc,
        "val_acc": validation_acc}