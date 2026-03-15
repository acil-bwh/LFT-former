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

def combination(how,num_slices,which,project_name,csv_match,axis):

    # --- Load Train/Validation Features ---

    if num_slices == 9:
        if axis == "coronal":
            raise ValueError(f"Can only be the original axial model")
        else:
            num_slices = num_slices
    
    elif num_slices == 20:
        if axis == "coronal":
            axis_suffix = "-cor"
        elif axis == "axial":
            axis_suffix = ""
    
    train1 = safe_load(f"{project_name}1-features-{num_slices}{axis_suffix}/train_features.npy")
    train2 = safe_load(f"{project_name}2-features-{num_slices}{axis_suffix}/train_features.npy")
    train3 = safe_load(f"{project_name}3-features-{num_slices}{axis_suffix}/train_features.npy")

    val1 = safe_load(f"{project_name}1-features-{num_slices}{axis_suffix}/valid_features.npy")
    val2 = safe_load(f"{project_name}2-features-{num_slices}{axis_suffix}/valid_features.npy")
    val3 = safe_load(f"{project_name}3-features-{num_slices}{axis_suffix}/valid_features.npy")

    test_1 = safe_load(f"{project_name}1-features-{num_slices}{axis_suffix}/test_features.npy")
    test_2 = safe_load(f"{project_name}2-features-{num_slices}{axis_suffix}/test_features.npy")
    test_3 = safe_load(f"{project_name}3-features-{num_slices}{axis_suffix}/test_features.npy")

    labtrain1 = safe_load(f"{project_name}1-features-{num_slices}{axis_suffix}/train_labels.npy")
    labtrain2 = safe_load(f"{project_name}2-features-{num_slices}{axis_suffix}/train_labels.npy")
    labtrain3 = safe_load(f"{project_name}3-features-{num_slices}{axis_suffix}/train_labels.npy")

    labval1 = safe_load(f"{project_name}1-features-{num_slices}{axis_suffix}/valid_labels.npy")
    labval2 = safe_load(f"{project_name}2-features-{num_slices}{axis_suffix}/valid_labels.npy")
    labval3 = safe_load(f"{project_name}3-features-{num_slices}{axis_suffix}/valid_labels.npy")

    labtest_1 = safe_load(f"{project_name}1-features-{num_slices}{axis_suffix}/test_labels.npy")
    labtest_2 = safe_load(f"{project_name}2-features-{num_slices}{axis_suffix}/test_labels.npy")
    labtest_3 = safe_load(f"{project_name}3-features-{num_slices}{axis_suffix}/test_labels.npy")

    # --- Concatenate bc train and val are separate features/files but same file ---

    train_1 = None
    if train1 is not None and val1 is not None:
        train_1 = np.concatenate([train1, val1], axis=0)
        print(f"train_1 created with shape: {train_1.shape}")
    else:
        print("Skipping train_1: train1 or val1 is missing.")

    train_2 = None
    if train2 is not None and val2 is not None:
        train_2 = np.concatenate([train2, val2], axis=0)
        print(f"train_2 created with shape: {train_2.shape}")
    else:
        print("Skipping train_2: train2 or val2 is missing.")

    train_3 = None
    if train3 is not None and val3 is not None:
        train_3 = np.concatenate([train3, val3], axis=0)
        print(f"train_3 created with shape: {train_3.shape}")
    else:
        print("Skipping train_3: train3 or val3 is missing.")

    ##############################################################

    labtrain_1 = None
    if labtrain1 is not None and labval1 is not None:
        labtrain_1 = np.concatenate([labtrain1, labval1], axis=0)
        print(f"labels train_1 created with shape: {labtrain_1.shape}")
    else:
        print("Skipping labels train_1: labtrain1 or labval1 is missing.")

    labtrain_2 = None
    if labtrain2 is not None and labval2 is not None:
        labtrain_2 = np.concatenate([labtrain2, labval2], axis=0)
        print(f"labels train_2 created with shape: {labtrain_2.shape}")
    else:
        print("Skipping labels train_2: labtrain2 or labval2 is missing.")

    labtrain_3 = None
    if labtrain3 is not None and labval3 is not None:
        labtrain_3 = np.concatenate([labtrain3, labval3], axis=0)
        print(f"labels train_3 created with shape: {labtrain_3.shape}")
    else:
        print("Skipping labels train_3: labtrain3 or labval3 is missing.")

    # --- Define dicts for calling tasks ---

    T = {
        "E": train_1,
        "T": train_2,
        "C": train_3}

    V = {
        "E": test_1,
        "T": test_2,
        "C": test_3}
    
    labT = {
        "E": labtrain_1,
        "T": labtrain_2,
        "C": labtrain_3}

    labV = {
        "E": labtest_1,
        "T": labtest_2,
        "C": labtest_3}
    
    print("\nStarting index matching and filtering (BEFORE combination)...")

    indices_to_keep_T = slice(None)
    indices_to_keep_V = slice(None)

    if which in ["ET","CT","ETC"]:
        try:
            match_indices_T = pd.read_csv(csv_match + "train.csv")
            match_indices_V = pd.read_csv(csv_match + "val.csv")
            
            original_indices_T = match_indices_T["idx_copd"].values
            original_indices_V = match_indices_V["idx_copd"].values
                        
            indices_to_keep_T = original_indices_T-1
            indices_to_keep_V = original_indices_V-1
            
        except Exception as e:
            print(f"[ERROR] Failed to load or adjust matching indices from CSV: {e}")
            print("Defaulting to keeping all indices.")
            indices_to_keep_T = slice(None)
            indices_to_keep_V = slice(None)
                
    original_shapes_T = {k: arr.shape for k, arr in T.items() if arr is not None}
    original_shapes_V = {k: arr.shape for k, arr in V.items() if arr is not None}
    original_shapes_labT = {k: arr.shape for k, arr in labT.items() if arr is not None}
    original_shapes_labV = {k: arr.shape for k, arr in labV.items() if arr is not None}

    print("Original features T shapes:", original_shapes_T)
    print(f"Original labels T shapes: {original_shapes_labT}")
    print("Original features V shapes:", original_shapes_V)
    print(f"Original labels V shapes: {original_shapes_labV}\n")

    # Apply filtering for E and C modalities features only bc of size mismatch with T
    for m in T:
        if T[m] is not None:
            if m in ["E", "C"] and indices_to_keep_T is not slice(None):
                T[m] = T[m][indices_to_keep_T]
                print(f"Filtered Train[{m}] features: {original_shapes_T.get(m)} -> {T[m].shape}")
            else:
                print(f"Kept Train[{m}] features: {T[m].shape}") # T[m].shape used for consistency

    for m in V:
        if V[m] is not None:
            if m in ["E", "C"] and indices_to_keep_V is not slice(None):
                V[m] = V[m][indices_to_keep_V]
                print(f"Filtered Val[{m}] features: {original_shapes_V.get(m)} -> {V[m].shape}")
            else:
                print(f"Kept Val[{m}] features: {V[m].shape}")

    for m in labT:
        if labT[m] is not None:
            if m in ["E", "C"] and indices_to_keep_T is not slice(None):
                labT[m] = labT[m][indices_to_keep_T]
                print(f"Filtered Train[{m}] labels: {original_shapes_labT.get(m)} -> {labT[m].shape}")
            else:
                print(f"Kept Train[{m}] labels: {labT[m].shape}") # T[m].shape used for consistency

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

    list_which_T = []
    list_which_V = []
    list_labels_T = []
    list_labels_V = []
    valid_weights = []

    for i, m in enumerate(requested):
        if T[m] is not None and V[m] is not None:
            list_which_T.append(T[m])
            list_which_V.append(V[m])
            list_labels_T.append(labT[m]) 
            list_labels_V.append(labV[m])
            valid_weights.append(weights[i])
        else:
            print(f"[WARNING] Skipping modality '{m}' because data is missing.")

    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()

    print(f"Combination = {how} for {which}",flush=True)
    print(f"Using modalities: {requested}",flush=True)
    print(f"Active modalities: {len(list_which_T)}",flush=True)
    print(f"Fusion weights: {valid_weights}",flush=True)

    if how == "concat":
        combined_featuresT = np.concatenate(list_which_T, axis=-1)
        combined_featuresV = np.concatenate(list_which_V, axis=-1)
    elif how == "mean":
        combined_featuresT = np.mean(list_which_T, axis=0)
        combined_featuresV = np.mean(list_which_V, axis=0)
    elif how == "min":
        combined_featuresT = np.minimum.reduce(list_which_T)
        combined_featuresV = np.minimum.reduce(list_which_V)
    elif how == "max":
        combined_featuresT = np.maximum.reduce(list_which_T)
        combined_featuresV = np.maximum.reduce(list_which_V)
    elif how == "weighted":
        weights = weights / weights.sum()
        combined_featuresT = np.tensordot(weights, np.stack(list_which_T), axes=(0,0))
        combined_featuresV = np.tensordot(weights, np.stack(list_which_V), axes=(0,0))
    elif how == "var":
        combined_featuresT = np.var(np.stack(list_which_T), axis=0)
        combined_featuresV = np.var(np.stack(list_which_V), axis=0)
    elif how == "std":
        combined_featuresT = np.std(np.stack(list_which_T), axis=0)
        combined_featuresV = np.std(np.stack(list_which_V), axis=0)
    elif how == "mix":
        stacked = np.stack(list_which_T, axis=-2)
        combined_featuresT = np.concatenate([np.mean(stacked, axis=-2), np.std(stacked, axis=-2)], axis=-1)
        stacked = np.stack(list_which_V, axis=-2)
        combined_featuresV = np.concatenate([np.mean(stacked, axis=-2), np.std(stacked, axis=-2)], axis=-1)
    elif how == "pca":
        from sklearn.decomposition import PCA
        n_patients_T, n_slices, n_augs, dim = list_which_T[0].shape
        stacked_T = np.concatenate(list_which_T, axis=-1)
        flat_T = stacked_T.reshape(-1, stacked_T.shape[-1])
        pca_T = PCA(n_components=dim)
        merged_flat_T = pca_T.fit_transform(flat_T)
        combined_featuresT = merged_flat_T.reshape(n_patients_T, n_slices, n_augs, dim)
        n_patients_V, n_slices, n_augs, dim = list_which_V[0].shape
        stacked_V = np.concatenate(list_which_V, axis=-1)
        flat_V = stacked_V.reshape(-1, stacked_V.shape[-1])
        pca_V = PCA(n_components=dim)
        merged_flat_V = pca_V.fit_transform(flat_V)
        combined_featuresV = merged_flat_V.reshape(n_patients_V, n_slices, n_augs, dim)
    else:
        raise ValueError(f"Aggregation method is not valid: {how}")
    
    if list_labels_T:
        combined_labelsT = list_labels_T[0]
    else:
        combined_labelsT = None

    if list_labels_V:
        combined_labelsV = list_labels_V[0] 
    else:
        combined_labelsV = None

    print("Combined T features shape:", combined_featuresT.shape,flush=True)
    print("Combined V features shape:", combined_featuresV.shape,flush=True)
    
    if combined_labelsT is not None:
        print("Combined T labels shape:", combined_labelsT.shape,flush=True)
        if combined_labelsT.ndim == 2 and combined_labelsT.shape[1] == 3:
             print(f"Labels kept shape (B, 3): {combined_labelsT.shape}")

    if combined_labelsV is not None:
        print("Combined V labels shape:", combined_labelsV.shape,flush=True)
        if combined_labelsV.ndim == 2 and combined_labelsV.shape[1] == 3:
             print(f"Labels kept shape (B, 3): {combined_labelsV.shape}")

    os.makedirs(f"{project_name}-features-{num_slices}{axis_suffix}", exist_ok=True)

    np.save(f"{project_name}-features-{num_slices}{axis_suffix}/{how}_train_features_{which}.npy", combined_featuresT)
    np.save(f"{project_name}-features-{num_slices}{axis_suffix}/{how}_train_labels_{which}.npy", combined_labelsT)
    np.save(f"{project_name}-features-{num_slices}{axis_suffix}/{how}_test_features_{which}.npy", combined_featuresV)
    np.save(f"{project_name}-features-{num_slices}{axis_suffix}/{how}_test_labels_{which}.npy", combined_labelsV)

    print(f"Combined T and V features using {how} for task(s) {which} saved!\n",flush=True)
    return

csv_match =  f"{project_name}-files/matching_indices_"

for axis in ["coronal","axial"]:
    if axis == "coronal":
        num_slices_list = [20]
    elif axis == "axial":
        num_slices_list = [9,20]
    for num_slices in num_slices_list:
        print(f"\n--- Combining for {num_slices} slices in {axis} ---\n")
        for how in ["concat","mean","weighted","max","min","var","std","mix","pca"]:
            for which in ["E","T","C","ET","CT","EC","ETC"]:
                combination(how,num_slices,which,project_name,csv_match,axis)

print("Multi-label and feature extraction finished for 9 and 20 slices using multiple embeddings and fusion methods")