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

class FeatureDataset(Dataset):
    def __init__(self, features_path, csv_path, dict_out, normalization=True):

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
        """Standardizes and bins labels based on the specific medical thresholds."""
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

# ---------------- PREDICTOR FUNCTION ----------------
def Predictor_mOutViT(main_path,
                   num_slices,
                   cuda_id,
                   dict_out,
                   task_dict,
                   how,
                   project_name,
                   input_type):

    # ---------------- PATHS ----------------
    cuda_set = f"cuda:{cuda_id}"
    save_path = os.path.join(main_path, f"{project_name}-results/models")

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(cuda_set if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

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

    csv_path = os.path.join(main_path, f"{project_name}-files",f"df_val_{model_name0}.csv")
    valid_features_path = os.path.join(f"{features_dir}-{num_slices}", f"{how}_test_features_{input_type}.npy")
    valid_dataset = FeatureDataset(valid_features_path, csv_path, dict_out, normalization=False)
    test_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    tr_path = os.path.join(main_path, f"{project_name}-files",f"df_train_{model_name0}.csv")
    train_features_path = os.path.join(f"{features_dir}-{num_slices}", f"{how}_train_features_{input_type}.npy")
    full_dataset = FeatureDataset(train_features_path, tr_path, dict_out, normalization=False)
    print("\nApplying normalization based on training set statistics...")    
    training_norm_params = {}
    for i, col_name in enumerate(full_dataset.task_columns_ordered):
        if task_dict[col_name]["type"] == "regression":
            train_data = full_dataset.labels[:, i]
            mask = ~np.isnan(train_data)
            training_norm_params[i] = {
                'mean': torch.tensor(np.nanmean(train_data[mask]), device=device),
                'std': torch.tensor(np.nanstd(train_data[mask]) + 1e-8, device=device)}
    
    # ---------------- MODEL LOADING ----------------
    from Utils.models.mOut_transformer import mOut_transformer

    model = mOut_transformer(
        num_vectors=num_slices,
        vec_dim=vec_dim,
        dict_out=dict_out,
        dim=256,
        depth=8,
        heads=4,
        mlp_dim=512,
        pool="cls").to(device)
    
    checkpoint_path = os.path.join(main_checkpoint_path, f"best_mOutViT_{fusion_type}-{how}_{input_type}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    num_classes_per_task = [len(v) if isinstance(v, list) else 1 for v in dict_out.values()]
    task_keys = list(dict_out.keys())
    n_samples = len(valid_dataset)
    
    task_priors = []
    labels_np = np.array(valid_dataset.labels)
    for i, n_classes in enumerate(num_classes_per_task):
        if n_classes > 1:
            valid_labels = labels_np[:, i].astype(int)
            valid_labels = valid_labels[valid_labels >= 0] # Filter NaNs/Missing
            counts = np.bincount(valid_labels, minlength=n_classes)
            priors = (counts + 1) / (counts.sum() + n_classes) # Laplacian smoothing
            task_priors.append(torch.tensor(priors, dtype=torch.float32, device=device))
        else:
            task_priors.append(None)

    real_labels = np.zeros((n_samples, len(task_keys)))
    pred_labels = np.zeros((n_samples, len(task_keys)))
    prob_storage = [np.zeros((n_samples, n_classes)) for n_classes in num_classes_per_task]

    # ---------------- INFERENCE LOOP ----------------
    print(f"Running inference on {n_samples} samples...")

    with torch.no_grad():
        for ix, (features, labels_tuple) in enumerate(tqdm(test_loader)):
            features = features.to(device)
            outputs = model(features)

            for t_idx, task_name in enumerate(task_keys):
                # Retrieve logits from the specific head
                logits = outputs[f"task{task_name}"]

                # Store Ground Truth
                real_val = labels_tuple[t_idx].item()
                real_labels[ix, t_idx] = real_val

                # Logic for Classification vs Regression
                if num_classes_per_task[t_idx] == 1:
                    # REGRESSION
                    pred_val = logits.item()
                    pred_labels[ix, t_idx] = pred_val
                    prob_storage[t_idx][ix, 0] = pred_val
                
                else:
                    # CLASSIFICATION
                    if task_priors[t_idx] is not None:
                        adjusted_logits = logits + torch.log(task_priors[t_idx] + 1e-8)
                        probs = torch.softmax(adjusted_logits, dim=-1)
                    else:
                        probs = torch.softmax(logits, dim=-1)
                    
                    pred = torch.argmax(probs, dim=1).item()
                    
                    # Store
                    pred_labels[ix, t_idx] = pred
                    prob_storage[t_idx][ix, :] = probs.cpu().numpy().flatten()

    # ---------------- 6. SAVE RESULTS ----------------
    for t_idx, task_name in enumerate(task_keys):
        # Result CSV
        summary_csv = os.path.join(save_path, f"model_mOutViT_{task_name}_{fusion_type}_{how}_{input_type}.csv")
        df_summary = pd.DataFrame({
            "data": np.arange(n_samples),
            "label": real_labels[:, t_idx],
            "predicted": pred_labels[:, t_idx]})
        df_summary.to_csv(summary_csv, index=False)

        # Probability CSV (Classifications only)
        if num_classes_per_task[t_idx] > 1:
            probs_csv = os.path.join(save_path, f"probs_mOutViT_{task_name}_{fusion_type}_{how}_{input_type}.csv")
            # Adjustment allows for class labels like -1, 0, 1 etc.
            adj = task_dict[task_name].get("adjustment", 0)
            cols = [f"probs_class{int(c - adj)}" for c in range(num_classes_per_task[t_idx])]
            
            df_probs = pd.DataFrame(prob_storage[t_idx], columns=cols)
            df_probs.insert(0, "data", np.arange(n_samples))
            df_probs.insert(1, "label", real_labels[:, t_idx])
            df_probs.to_csv(probs_csv, index=False)

    print(f"\nInference finished. Results saved to {save_path}")
    return