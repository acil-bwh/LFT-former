# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path")
parser.add_argument("--input", type=str, help="E,C,T,EC,CT,ET,ETC",default="T")
parser.add_argument("--output", type=str, help="EMPH, TRAJ or COPD",default="TRAJ")
parser.add_argument("--how", type=str, help="how are weight features fused -- concat/mean/weighted/max/min/var/std/mix/pca/",default="concat")
parser.add_argument("--add", type=int, help="1: age, 2: age+gender, 3: age+gender+packs",default=0)
parser.add_argument("--project_name", type=str,default="COPDGene")
parser.add_argument("--slices", type=int, default=20)
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="modular")
args = parser.parse_args()

main_path = args.path
output_type = args.output
input_type = args.input
vars_add = args.add
how = args.how
project_name = args.project_name
slices = args.slices
wrapping_mode = args.wrap

if project_name == "ECLIPSE":
    proj_lab = "ECLIPSE"
else:
    proj_lab = "COPD"

# ---------------- PATHS ----------------
origin_path = os.path.join(main_path, project_name+'-results/models')
save_path = os.path.join(main_path, project_name+'-results/figures')
csv_path = os.path.join(main_path, project_name+'-results/metrics')
for p in [origin_path, save_path, csv_path]:
    os.makedirs(p, exist_ok=True)

if output_type == "EMPH":
    tasks = {1: {"prob_cols": ["probs_class0","probs_class1","probs_class2","probs_class3"], 
        "labels": {0:'None',1:'Mild',2:'Moderate',3:'High'}, "add": 0}}
elif output_type == "TRAJ":
    tasks = {1: {"prob_cols": ["probs_class1","probs_class2","probs_class3","probs_class4","probs_class5","probs_class6"], 
        "labels": {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}, "add": 1}}
elif output_type == "COPD":
    tasks = {1: {"prob_cols": ["probs_class-1","probs_class0","probs_class1","probs_class2","probs_class3","probs_class4"], 
        "labels": {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}, "add": -1},}

task_names = ["taskTRAJ"]
fusion_map = {9:'C', 20:'W'}
        
for sets in [1,2]:
    for model_id, task in tasks.items():
        prob_cols = task["prob_cols"]
        class_labels = task["labels"]
        add = task["add"]
        task_name = task_names[model_id-1]
        for num_slices in [20]:
            fusion_type = fusion_map[num_slices]
            csv_file = os.path.join(origin_path, f"probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set{sets}.csv")

            if not os.path.exists(csv_file):
                print(f"File not found, skipping: {csv_file}")
                continue

            df = pd.read_csv(csv_file)
            print(f"Loading: {csv_file}")

            y_true = df["label"].values - add
            y_score = df[prob_cols].values

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
            results_df.to_csv(os.path.join(csv_path, f"classAUC_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set{sets}.csv"), index=False)
            print(results_df.to_string(index=False))
            print(f"\nMacro-average AUC: {macro_auc:.4f}\n")

            # Plot ROC curves
            plt.figure(figsize=(8,6))
            for cls in classes:
                plt.plot(fpr[cls], tpr[cls], lw=2, label=f"{class_labels[cls]} (AUC={roc_auc[cls]:.3f})")
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlabel("False Positive Rate", fontsize=16)
            plt.ylabel("True Positive Rate", fontsize=16)
            plt.title(f"ROC (Macro AUC={macro_auc:.3f}) task: {output_type} with {input_type}", fontsize=16)
            plt.legend(loc="lower right", fontsize=14)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"rocAUC_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set{sets}.png"), dpi=300)
            plt.close()

print("Starting comparsion...")
for model_id, task in tasks.items():
    prob_cols = task["prob_cols"]
    class_labels = task["labels"]
    add = task["add"]
    task_name = task_names[model_id-1]
    for num_slices in [20]:
        fusion_type = fusion_map[num_slices]

        which_file1= origin_path+f"/probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set1.csv"
        which_file2= origin_path+f"/probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set2.csv"

        if not os.path.exists(which_file1):
            print(f"File not found, skipping: {which_file1}")
            continue
        if not os.path.exists(which_file2):
            print(f"File not found, skipping: {which_file2}")
            continue  
    
        if os.path.exists(which_file1) and os.path.exists(which_file2):
            print(f"Loading: {which_file1} and {which_file2}")

            df1 = pd.read_csv(which_file1)
            df1 = df1.iloc[:,1:]
            df2 = pd.read_csv(which_file2)
            df2 = df2.iloc[:,1:]

            classes = np.array(list(class_labels.keys()))

            true_labels = df1["label"].values - add
            y_score1 = df1[prob_cols].values
            y_score2 = df2[prob_cols].values

            y_true_bin = label_binarize(true_labels, classes=classes)

            fpr1, tpr1, roc_auc1 = {}, {}, {}
            fpr2, tpr2, roc_auc2 = {}, {}, {}

            for i, cls in enumerate(classes):
                y_true_cls = y_true_bin[:, i]
                
                if len(np.unique(y_true_cls)) < 2:  # skip if not enough samples
                    roc_auc1[cls] = np.nan
                    roc_auc2[cls] = np.nan
                    continue
                
                fpr1[cls], tpr1[cls], _ = roc_curve(y_true_cls, y_score1[:, i])
                roc_auc1[cls] = auc(fpr1[cls], tpr1[cls])
                
                fpr2[cls], tpr2[cls], _ = roc_curve(y_true_cls, y_score2[:, i])
                roc_auc2[cls] = auc(fpr2[cls], tpr2[cls])

            results = pd.DataFrame({
                "Class": [class_labels[c] for c in classes],
                f"AUC_{proj_lab}1": [roc_auc1[c] for c in classes],
                f"AUC_{proj_lab}2": [roc_auc2[c] for c in classes]})
            results.to_csv(csv_path+f"/classAUC_linLFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_sets.csv", index=False)

            import matplotlib.cm as cm
            colors = cm.get_cmap('tab10', len(classes))  # tab10 palette for up to 10 classes

            plt.figure(figsize=(10,7))
            for i, cls in enumerate(classes):
                if np.isnan(roc_auc1[cls]) or np.isnan(roc_auc2[cls]):
                    continue
                color = colors(i)
                plt.plot(fpr1[cls], tpr1[cls], lw=2, color=color,
                        label=f"{class_labels[cls]} {proj_lab}1 (AUC={roc_auc1[cls]:.3f})")
                plt.plot(fpr2[cls], tpr2[cls], lw=2, linestyle='--', color=color,
                        label=f"{class_labels[cls]} {proj_lab}2 (AUC={roc_auc2[cls]:.3f})")
                
            plt.plot([0,1], [0,1], 'k--', lw=1)
            plt.xlabel("False Positive Rate",fontsize=16)
            plt.ylabel("True Positive Rate",fontsize=16)
            plt.title(f"One-vs-All ROC Curves COPD1-vs-COPD2",fontsize=16)
            plt.legend(loc="lower right", fontsize=14)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(save_path+f"/rocAUC_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_sets.png", dpi=300)
            plt.show()

            print('')
            print("Results for model "+str(model_id)+" with "+str(num_slices)+"-slices")
            print('')
            print(results.to_string(index=False))
            # === Macro AUC comparison ===
            macro_auc1 = roc_auc_score(y_true_bin, y_score1, average='weighted',multi_class='ovr')
            macro_auc2 = roc_auc_score(y_true_bin, y_score2, average='weighted',multi_class='ovr')
            print('')
            print(f"Macro AUC {proj_lab}1: {macro_auc1:.4f}")
            print(f"Macro AUC {proj_lab}2: {macro_auc2:.4f}")

print("Finished comparsion")
