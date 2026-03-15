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
num_slices = 20

def safe_load(path):
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"[WARNING] Missing file: {path}")
        return None

def aggregate(data_list, method, weights_dict=None, requested_keys=None):
    """Generic aggregation function to handle different fusion methods."""
    if not data_list:
        return None
    
    if method == "concat":
        return np.concatenate(data_list, axis=-1)
    elif method == "mean":
        return np.mean(data_list, axis=0)
    elif method == "max":
        return np.maximum.reduce(data_list)
    elif method == "min":
        return np.minimum.reduce(data_list)
    elif method == "var":
        return np.var(data_list, axis=0)
    elif method == "std":
        return np.std(data_list, axis=0)
    elif method == "weighted" and weights_dict and requested_keys:
        w = np.array([weights_dict[m] for m in requested_keys])
        w /= w.sum()
        return np.tensordot(w, np.stack(data_list), axes=(0,0))
    elif method == "mix":
        stacked = np.stack(data_list)
        return np.concatenate([np.mean(stacked, axis=0), np.std(stacked, axis=0)], axis=-1)
    return data_list[0] # Default fallback

def combination(view_how, mod_how, which, project_name, csv_match):
    print(f"\n--- Fusion: View({view_how}) + Modality({mod_how}) for {which} ---", flush=True)
    
    views = {"ax": "", "cor": "-cor"}
    modalities = {"E": "1", "T": "2", "C": "3"}
    
    raw_data_T = {"ax": {}, "cor": {}}
    raw_data_V = {"ax": {}, "cor": {}}
    raw_labs_T = {"ax": {}}
    raw_labs_V = {"ax": {}}

    # 1. LOAD AND PREPARE
    for v_key, v_suffix in views.items():
        for m_key, m_num in modalities.items():
            base_path = f"{project_name}{m_num}-features-{num_slices}{v_suffix}"
            
            f_train, f_valid, f_test = safe_load(f"{base_path}/train_features.npy"), \
                                       safe_load(f"{base_path}/valid_features.npy"), \
                                       safe_load(f"{base_path}/test_features.npy")
            l_train, l_valid, l_test = safe_load(f"{base_path}/train_labels.npy"), \
                                       safe_load(f"{base_path}/valid_labels.npy"), \
                                       safe_load(f"{base_path}/test_labels.npy")

            if f_train is not None and f_valid is not None:
                raw_data_T[v_key][m_key] = np.concatenate([f_train, f_valid], axis=0)
                if v_key == "ax": raw_labs_T["ax"][m_key] = np.concatenate([l_train, l_valid], axis=0)
            
            raw_data_V[v_key][m_key] = f_test
            if v_key == "ax": raw_labs_V["ax"][m_key] = l_test

    # 2. FILTERING (Consistency with T modality)
    if which in ["ET", "CT", "ETC"]:
        try:
            idx_T = pd.read_csv(csv_match + "train.csv")["idx_copd"].values - 1
            idx_V = pd.read_csv(csv_match + "val.csv")["idx_copd"].values - 1
            for v in views:
                for m in ["E", "C"]:
                    if raw_data_T[v][m] is not None: raw_data_T[v][m] = raw_data_T[v][m][idx_T]
                    if raw_data_V[v][m] is not None: raw_data_V[v][m] = raw_data_V[v][m][idx_V]
            for m in ["E", "C"]:
                if m in raw_labs_T["ax"]: raw_labs_T["ax"][m] = raw_labs_T["ax"][m][idx_T]
                if m in raw_labs_V["ax"]: raw_labs_V["ax"][m] = raw_labs_V["ax"][m][idx_V]
        
        except Exception as e: print(f"Filter error: {e}")

    # 3. STEP 1: VIEW FUSION (Axial + Coronal)
    fused_view_T = {}
    fused_view_V = {}
    for m in modalities:
        list_v_T = [raw_data_T[v][m] for v in views if raw_data_T[v][m] is not None]
        list_v_V = [raw_data_V[v][m] for v in views if raw_data_V[v][m] is not None]
        fused_view_T[m] = aggregate(list_v_T, view_how)
        fused_view_V[m] = aggregate(list_v_V, view_how)

    # 4. STEP 2: MODALITY FUSION (E, T, C)
    modalities_map = {"E": ["E"], "T": ["T"], "C": ["C"], "ET": ["E", "T"], "CT": ["T", "C"], "ETC": ["E", "T", "C"]}
    requested = modalities_map[which]
    
    list_m_T = [fused_view_T[m] for m in requested if fused_view_T[m] is not None]
    list_m_V = [fused_view_V[m] for m in requested if fused_view_V[m] is not None]

    if mod_how == "pca":
        from sklearn.decomposition import PCA
        # Concat first to provide all variance to PCA
        concat_T = np.concatenate(list_m_T, axis=-1)
        orig_shape = concat_T.shape
        flat_T = concat_T.reshape(orig_shape[0], -1)
        pca = PCA(n_components=min(orig_shape[0], 512))
        combined_T = pca.fit_transform(flat_T)
        
        concat_V = np.concatenate(list_m_V, axis=-1)
        combined_V = pca.transform(concat_V.reshape(concat_V.shape[0], -1))
    else:
        weights = {"E": 0.4, "T": 0.2, "C": 0.4}
        combined_T = aggregate(list_m_T, mod_how, weights, requested)
        combined_V = aggregate(list_m_V, mod_how, weights, requested)

    # 5. SAVE
    out_dir = f"{project_name}-features-20-mv"
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"V-{view_how}_M-{mod_how}"
    
    np.save(f"{out_dir}/{prefix}_train_features_{which}.npy", combined_T)
    np.save(f"{out_dir}/{prefix}_train_labels_{which}.npy", raw_labs_T["ax"][requested[0]])
    np.save(f"{out_dir}/{prefix}_test_features_{which}.npy", combined_V)
    np.save(f"{out_dir}/{prefix}_test_labels_{which}.npy", raw_labs_V["ax"][requested[0]])
    
    print(f"Success! Final shape: {combined_T.shape}", flush=True)

csv_path = f"{project_name}-files/matching_indices_"

# view_methods = ["mean", "concat", "max"]
# mod_methods = ["concat", "mean", "weighted", "max", "pca"]

view_methods = ["max"]
mod_methods = ["concat", "mean", "weighted", "max"]
tasks = ["E", "T", "C", "ET", "CT", "ETC"]

for v_h in view_methods:
    for m_h in mod_methods:
        for t in tasks:
            combination(v_h, m_h, t, project_name, csv_path)

print('Done')