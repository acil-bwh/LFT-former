# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································
import numpy as np
import os
import argparse
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, default="COPDGene")

args = parser.parse_args()

project_name = args.project_name

def safe_load(path):
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"[WARNING] Missing file: {path}")
        return None

def combination(how,num_slices,which,project_name,csv_match,set_num):

    # --- Load Validation Features and Labels ---

    test_1 = safe_load(f"{project_name}1-features_sets-{num_slices}/test_features_set{set_num}.npy")
    test_2 = safe_load(f"{project_name}2-features_sets-{num_slices}/test_features_set{set_num}.npy")
    test_3 = safe_load(f"{project_name}3-features_sets-{num_slices}/test_features_set{set_num}.npy")

    labtest_1 = safe_load(f"{project_name}1-features_sets-{num_slices}/test_labels_set{set_num}.npy")
    labtest_2 = safe_load(f"{project_name}2-features_sets-{num_slices}/test_labels_set{set_num}.npy")
    labtest_3 = safe_load(f"{project_name}3-features_sets-{num_slices}/test_labels_set{set_num}.npy")

    # --- Define dicts for calling tasks ---

    V = {
        "E": test_1,
        "T": test_2,
        "C": test_3}
    
    labV = {
        "E": labtest_1,
        "T": labtest_2,
        "C": labtest_3}
    
    print("\nStarting index matching and filtering (BEFORE combination)...")

    indices_to_keep_V = slice(None)

    if which in ["ET","CT","ETC"]:
        try:
            match_indices_V = pd.read_csv(csv_match)
            original_indices_V = match_indices_V["idx_copd"].values
            indices_to_keep_V = original_indices_V-1
            
        except Exception as e:
            print(f"[ERROR] Failed to load or adjust matching indices from CSV: {e}")
            print("Defaulting to keeping all indices.")
            indices_to_keep_V = slice(None)
                
    original_shapes_V = {k: arr.shape for k, arr in V.items() if arr is not None}
    original_shapes_labV = {k: arr.shape for k, arr in labV.items() if arr is not None}

    print("Original features V shapes:", original_shapes_V)
    print(f"Original labels V shapes: {original_shapes_labV}\n")

    # Apply filtering for E and C modalities features only bc of size mismatch with T
    for m in V:
        if V[m] is not None:
            if m in ["E", "C"] and indices_to_keep_V is not slice(None):
                V[m] = V[m][indices_to_keep_V]
                print(f"Filtered Val[{m}] features: {original_shapes_V.get(m)} -> {V[m].shape}")
            else:
                print(f"Kept Val[{m}] features: {V[m].shape}")

    for m in labV:
        if labV[m] is not None:
            if m in ["E", "C"] and indices_to_keep_V is not slice(None):
                labV[m] = labV[m][indices_to_keep_V]
                print(f"Filtered Val[{m}] labels: {original_shapes_labV.get(m)} -> {labV[m].shape}")
            else:
                print(f"Kept Val[{m}] labels: {labV[m].shape}")
    
    # --- Modality Selection and Combination (Fusion) ---
    modalities_map = {"E":  ["E"], "T":  ["T"], "C":  ["C"], "ET": ["E", "T"],
                      "EC": ["E", "C"], "CT": ["T", "C"], "ETC": ["E", "T", "C"]}
    default_weights = {"E":   [1.0], "T":   [1.0], "C":   [1.0], "ET":  [0.3, 0.7],
                       "EC":  [0.5, 0.5], "CT":  [0.5, 0.5], "ETC": [0.4, 0.2, 0.4]}

    requested = modalities_map[which]
    weights   = np.array(default_weights[which], dtype=float)

    list_which_V = []
    list_labels_V = []
    valid_weights = []

    for i, m in enumerate(requested):
        if V[m] is not None:
            list_which_V.append(V[m])
            list_labels_V.append(labV[m])
            valid_weights.append(weights[i])
        else:
            print(f"[WARNING] Skipping modality '{m}' because data is missing.")

    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()

    print(f"Combination = {how} for {which}",flush=True)
    print(f"Using modalities: {requested}",flush=True)
    print(f"Active modalities: {len(list_which_V)}",flush=True)
    print(f"Fusion weights: {valid_weights}",flush=True)

    if how == "concat":
        combined_featuresV = np.concatenate(list_which_V, axis=-1)
    elif how == "mean":
        combined_featuresV = np.mean(list_which_V, axis=0)
    elif how == "min":
        combined_featuresV = np.minimum.reduce(list_which_V)
    elif how == "max":
        combined_featuresV = np.maximum.reduce(list_which_V)
    elif how == "weighted":
        weights = weights / weights.sum()
        combined_featuresV = np.tensordot(weights, np.stack(list_which_V), axes=(0,0))
    elif how == "var":
        combined_featuresV = np.var(np.stack(list_which_V), axis=0)
    elif how == "std":
        combined_featuresV = np.std(np.stack(list_which_V), axis=0)
    elif how == "mix":
        stacked = np.stack(list_which_V, axis=-2)
        combined_featuresV = np.concatenate([np.mean(stacked, axis=-2), np.std(stacked, axis=-2)], axis=-1)
    elif how == "pca":
        from sklearn.decomposition import PCA
        n_patients_V, n_slices, n_augs, dim = list_which_V[0].shape
        stacked_V = np.concatenate(list_which_V, axis=-1)
        flat_V = stacked_V.reshape(-1, stacked_V.shape[-1])
        pca_V = PCA(n_components=dim)
        merged_flat_V = pca_V.fit_transform(flat_V)
        combined_featuresV = merged_flat_V.reshape(n_patients_V, n_slices, n_augs, dim)
    else:
        raise ValueError(f"Aggregation method is not valid: {how}")

    if list_labels_V:
        combined_labelsV = list_labels_V[0] 
    else:
        combined_labelsV = None
        
    print("Combined V features shape:", combined_featuresV.shape,flush=True)
    
    if combined_labelsV is not None:
        print("Combined V labels shape:", combined_labelsV.shape,flush=True)
        if combined_labelsV.ndim == 2 and combined_labelsV.shape[1] == 3:
             print(f"Labels kept shape (B, 3): {combined_labelsV.shape}")

    os.makedirs(f"{project_name}-features_sets-{num_slices}", exist_ok=True)

    np.save(f"{project_name}-features_sets-{num_slices}/{how}_test_features_{which}_set{set_num}.npy", combined_featuresV)
    np.save(f"{project_name}-features_sets-{num_slices}/{how}_test_labels_{which}_set{set_num}.npy", combined_labelsV)

    print(f"Combined V features using {how} for task(s) {which} saved!\n",flush=True)
    return

csv_match =  f"{project_name}files/matching_indices_sets.csv"
for set_num in [1,2]:
    for num_slices in [9,20]:
        print(f"\n--- Combining for {num_slices} slices --- set {set_num} \n")
        for how in ["concat","mean","weighted","max","min","var","std","mix","pca"]:
            for which in ["E","T","C","ET","CT","EC","ETC"]:
                combination(how,num_slices,which,project_name,csv_match,set_num)

print("Multi-label and feature extraction finished for 9 and 20 slices using multiple embeddings and fusion methods using both sets")