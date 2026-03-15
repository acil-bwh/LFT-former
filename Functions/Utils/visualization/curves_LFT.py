# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path")
parser.add_argument("--add", type=int, help="1: age, 2: age+gender, 3: age+gender+packs",default=0)
parser.add_argument("--project_name", type=str,default="COPDGene")
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="modular")

args = parser.parse_args()

main_path = args.path
vars_add = args.add
project_name = args.project_name
wrapping_mode = args.wrap

# ---------------- PATHS ----------------
origin_path = os.path.join(main_path, project_name+'-results/models')
save_path = os.path.join(main_path, project_name+'-results/figures')
csv_path = os.path.join(main_path, project_name+'-results/metrics')
for p in [origin_path, save_path, csv_path]:
    os.makedirs(p, exist_ok=True)

tasks = {"prob_cols": ["probs_class1","probs_class2","probs_class3","probs_class4","probs_class5","probs_class6"], 
    "labels": {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}}
add = -1

# ---------------- ROC/AUC PLOTTING ----------------
for task in tasks.items():
    prob_cols = task["prob_cols"]
    class_labels = task["labels"]
    csv_file = os.path.join(origin_path, f"probs_LFT_{vars_add}_{wrapping_mode}.csv")
    pred_file = os.path.join(origin_path, f"model_LFT_{vars_add}_{wrapping_mode}.csv")
    
    if not os.path.exists(csv_file):
        print(f"File not found, skipping: {csv_file}")
        continue

    df = pd.read_csv(csv_file)
    df = df.dropna()
    print(f"Loading: {csv_file}")

    y_true = df["label"].values - add
    y_score = df[prob_cols].values

    y_pred = pd.read_csv(pred_file)
    y_pred = y_pred.iloc[:,3] - add

    print("True: ",np.unique(y_true))
    print("Pred: ",np.unique(y_pred))

    classes = np.array(list(class_labels.keys()))
    y_true_bin = label_binarize(y_true, classes=classes)

    # Compute per-class ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i, cls in enumerate(classes):
        y_cls = y_true_bin[:, i]
        fpr[cls], tpr[cls], _ = roc_curve(y_cls, y_score[:, i])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    # Macro-average AUC
    macro_auc = roc_auc_score(y_true_bin, y_score, average='macro')

    # Save per-class AUC CSV
    results_df = pd.DataFrame({
        "Class": [class_labels[c] for c in classes],
        "AUC": [roc_auc[c] for c in classes]})
    
    results_df.to_csv(os.path.join(csv_path, f"classAUC_LFT_{vars_add}_{wrapping_mode}.csv"), index=False)
    
    print(results_df.to_string(index=False))
    print(f"\nMacro-average AUC: {macro_auc:.4f}\n")

    # Plot ROC curves
    plt.figure(figsize=(8,6))
    for cls in classes:
        plt.plot(fpr[cls], tpr[cls], lw=2, label=f"{class_labels[cls]} (AUC={roc_auc[cls]:.3f})")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title(f"ROC (Macro AUC={macro_auc:.3f})", fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"rocAUC_LFT_{vars_add}_{wrapping_mode}.png"), dpi=300)
    plt.close()
