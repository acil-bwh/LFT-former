# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2026 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

import os
import argparse
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, accuracy_score, auc, roc_curve,
                             cohen_kappa_score, balanced_accuracy_score, roc_auc_score)
from itertools import cycle

warnings.filterwarnings("ignore")

def preprocess_data(X):
    n_patients, n_slices, n_augs, dim = X.shape
    return X[:, :, 0, :].reshape(n_patients, -1)

def apply_cross_attention(Q, K, V):
    """NumPy implementation of Scaled Dot-Product Attention."""
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, V)

def calculate_one_off_metrics(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    one_off_acc = np.mean(diff <= 1)
    y_pred_adj = np.where(diff <= 1, y_true, y_pred)
    one_off_kappa = cohen_kappa_score(y_true, y_pred_adj)
    return one_off_acc, one_off_kappa

def run_experiment(clf_name, Z_train, Z_test, y_train, y_test, classes, results_path, view_name):
    os.makedirs(results_path, exist_ok=True)
    
    if clf_name == 'svm':
        clf = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
    elif clf_name == 'rf':
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)
    elif clf_name == 'xgb':
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, objective='multi:softprob')
    
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    y_score = clf.predict_proba(Z_test)
    
    weighted_kappa = cohen_kappa_score(y_test, y_pred, weights='linear')
    one_off_acc, _ = calculate_one_off_metrics(y_test, y_pred)
    
    res = {
        'View': view_name, 
        'Classifier': clf_name.upper(), 
        'F1-Macro': f1_score(y_test, y_pred, average='macro'), 
        'Accuracy': accuracy_score(y_test, y_pred),
        'Weighted_Kappa': weighted_kappa,
        'One_Off_Acc': one_off_acc
    }
    
    joblib.dump(clf, os.path.join(results_path, f"model_{clf_name}_{view_name}.pkl"))
    return res

def apply_gated_fusion(Z1, Z2, y):
    """
    Learns a gating mechanism to weight two views.
    Returns fused features for train and a gate model to transform test.
    """
    # Concatenate views to let the gate 'see' both
    gate_input = np.hstack([Z1, Z2])
    # We use a simple linear gate (Logistic Regression) to predict class importance
    # Here we simplify: the gate learns a weighting vector for the feature space
    gate = LogisticRegression(max_iter=500).fit(gate_input, y)
    
    # Get probability of the 'strongest' class as a proxy for confidence/weight
    # This acts as our alpha
    alpha = gate.predict_proba(gate_input)[:, 1].reshape(-1, 1) 
    Z_fused = alpha * Z1 + (1 - alpha) * Z2
    return Z_fused, gate

def Trainer_Complete_Comparison(main_path, num_slices, project_name, model_id, fusion_type):
    base_results = os.path.join(main_path, f"{project_name}-results/PCA-{model_id}", fusion_type)
    os.makedirs(base_results, exist_ok=True)
    
    added = [0, -1, 1]
    IT = {1: 0, 2: 1, 3: 2}.get(model_id, 1)
    feat_base = os.path.join(main_path, f"{project_name}{model_id}-features")
    
    # --- LOAD ---
    def load_set(name):
        ax = preprocess_data(np.load(os.path.join(f"{feat_base}-{num_slices}", f"{name}_features.npy")))
        cor = preprocess_data(np.load(os.path.join(f"{feat_base}-{num_slices}-cor", f"{name}_features.npy")))
        y = np.load(os.path.join(f"{feat_base}-{num_slices}", f"{name}_labels_{IT+1}.npy")).ravel()
        return ax, cor, y

    X_tr_ax, X_tr_cor, y_train = load_set("train")
    X_val_ax, X_val_cor, y_val = load_set("valid")
    X_te_ax, X_te_cor, y_test = load_set("test")

    X_tr_ax, X_tr_cor = np.vstack([X_tr_ax, X_val_ax]), np.vstack([X_tr_cor, X_val_cor])
    y_train = np.concatenate([y_train, y_val])

    if model_id == 3:
        y_train[y_train == -2], y_test[y_test == -2] = 0, 0
    y_test, y_train = (y_test + added[IT]).astype(int), (y_train + added[IT]).astype(int) 

    # --- INDIVIDUAL VIEW PCA ---
    sc_ax, sc_cor = StandardScaler().fit(X_tr_ax), StandardScaler().fit(X_tr_cor)
    Z_tr_ax = PCA(n_components=0.95).fit_transform(sc_ax.transform(X_tr_ax))
    Z_te_ax = PCA(n_components=0.95).fit(sc_ax.transform(X_tr_ax)).transform(sc_ax.transform(X_te_ax))
    
    Z_tr_cor = PCA(n_components=0.95).fit_transform(sc_cor.transform(X_tr_cor))
    Z_te_cor = PCA(n_components=0.95).fit(sc_cor.transform(X_tr_cor)).transform(sc_cor.transform(X_te_cor))

    # --- FUSION LOGIC ---
    if fusion_type == 'pca':
        # Joint PCA: Concat raw features -> Scaler -> PCA
        X_tr_joint = np.hstack([X_tr_ax, X_tr_cor])
        X_te_joint = np.hstack([X_te_ax, X_te_cor])
        sc_joint = StandardScaler().fit(X_tr_joint)
        pca_joint = PCA(n_components=0.95).fit(sc_joint.transform(X_tr_joint))
        Z_tr_fused = pca_joint.transform(sc_joint.transform(X_tr_joint))
        Z_te_fused = pca_joint.transform(sc_joint.transform(X_te_joint))
        
    elif fusion_type == 'concat':
        Z_tr_fused = np.hstack([Z_tr_ax, Z_tr_cor])
        Z_te_fused = np.hstack([Z_te_ax, Z_te_cor])
        
    else:
        # Dimensionality alignment for element-wise/statistical ops
        d = min(Z_tr_ax.shape[1], Z_tr_cor.shape[1])
        z_ax_tr, z_cor_tr = Z_tr_ax[:, :d], Z_tr_cor[:, :d]
        z_ax_te, z_cor_te = Z_te_ax[:, :d], Z_te_cor[:, :d]
        
        # Stack to shape (N, 2, D) to compute stats along the view axis (axis=1)
        stacked_tr = np.stack([z_ax_tr, z_cor_tr], axis=1)
        stacked_te = np.stack([z_ax_te, z_cor_te], axis=1)

        if fusion_type == 'gated':
            Z_tr_fused, gate_model = apply_gated_fusion(z_ax_tr, z_cor_tr, y_train)
            alpha_te = gate_model.predict_proba(np.hstack([z_ax_te, z_cor_te]))[:, 1].reshape(-1, 1)
            Z_te_fused = alpha_te * z_ax_te + (1 - alpha_te) * z_cor_te
        elif  fusion_type == 'mean':
            Z_tr_fused, Z_te_fused = np.mean(stacked_tr, axis=1), np.mean(stacked_te, axis=1)
        elif fusion_type == 'max':
            Z_tr_fused, Z_te_fused = np.max(stacked_tr, axis=1), np.max(stacked_te, axis=1)
        elif fusion_type == 'min':
            Z_tr_fused, Z_te_fused = np.min(stacked_tr, axis=1), np.min(stacked_te, axis=1)
        elif fusion_type == 'std':
            Z_tr_fused, Z_te_fused = np.std(stacked_tr, axis=1), np.std(stacked_te, axis=1)
        elif fusion_type == 'var':
            Z_tr_fused, Z_te_fused = np.var(stacked_tr, axis=1), np.var(stacked_te, axis=1)
        elif fusion_type == 'attention':
            Z_tr_fused = apply_cross_attention(z_ax_tr, z_cor_tr, z_cor_tr)
            Z_te_fused = apply_cross_attention(z_ax_te, z_cor_te, z_cor_te)

    view_data = {'Axial': (Z_tr_ax, Z_te_ax), 'Coronal': (Z_tr_cor, Z_te_cor), 'Fused': (Z_tr_fused, Z_te_fused)}
    comparison_results = []
    classes = np.sort(np.unique(y_train))

    for view_name, (train_data, test_data) in view_data.items():
        for clf_name in ['svm', 'rf', 'xgb']:
            res = run_experiment(clf_name, train_data, test_data, y_train, y_test, classes, 
                                 os.path.join(base_results, view_name, clf_name), view_name)
            comparison_results.append(res)

    # --- SUMMARY ---
    df = pd.DataFrame(comparison_results)
    df.to_csv(os.path.join(base_results, f"metrics_{fusion_type}.csv"), index=False)
    
    for metric in ["F1-Macro", "Weighted_Kappa"]:
        pivot = df.pivot(index="Classifier", columns="View", values=metric)
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title(f"{metric} ({fusion_type.upper()})")
        plt.savefig(os.path.join(base_results, f"heatmap_{metric}_{fusion_type}.png"))
        plt.close()
        
    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="COPDGene")
    parser.add_argument("--slices", type=int, default=20)
    parser.add_argument("--model", type=int, default=2)
    parser.add_argument("--fusion", type=str, default="concat", 
                        choices=['concat', 'mean', 'max', 'min', 'std', 'var', 'pca', 'attention', 'gated'],
                        help="Strategy to fuse Axial and Coronal embeddings")
    args = parser.parse_args()

    Trainer_Complete_Comparison(args.path, args.slices, args.project_name, args.model, args.fusion)