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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path")
parser.add_argument("--project_name", type=str, default="COPDGene")

args = parser.parse_args()

main_path = args.path
project_name = args.project_name

origin_path = f"{main_path}{project_name}-results/models"
save_path = f"{main_path}{project_name}-results/figures"
csv_path = f"{main_path}{project_name}-results/metrics"

import os
for paths in [origin_path,save_path,csv_path]:
    if not os.path.exists(paths):
        os.makedirs(paths)

for model_id in [1,2,3,4]:
    if model_id == 1:
        prob_cols = ["probs0", "probs1", "probs2", "probs3"]
        class_labels = {0:'None',1:'Mild',2:'Moderate',3:'High'}
        add = 0
    elif model_id == 2:
        prob_cols = ["probs1", "probs2", "probs3", "probs4", "probs5", "probs6"]
        class_labels = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
        add = +1
    elif model_id == 3:
        prob_cols = ["probs-1", "probs0", "probs1", "probs2", "probs3", "probs4"]
        class_labels = {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}
        add = -1
    elif model_id == 4:
        prob_cols = ["probs0", "probs1", "probs2", "probs3"]
        class_labels = {0:'Low',1:'Mild',2:'Moderate',3:'High'}
        add = 0
    else:
        raise Exception("Sorry, invalid model (must be 1, 2, 3 or 4)")

    for batch_size in [1,9,20]:
        if batch_size == 1:
            fusion_type = ''
        elif batch_size == 9:
            fusion_type = 'C'
        elif batch_size == 20:
            fusion_type = 'W'
        else:
            raise(Exception('Can only be 1, 9 or 20 slices'))   
    
        which_file = origin_path+"/probs_RegionViT_"+str(model_id)+fusion_type+".csv"

        if os.path.exists(which_file):
            df = pd.read_csv(which_file)
            print(f"Loading: {which_file}")
            df = df.dropna(subset=["label"])

            def bin_labels(val):
                if val <= 10:
                    return 0
                elif 10 < val <= 20:
                    return 1
                elif 20 < val <= 40:
                    return 2
                else: # > 40
                    return 3
            
            if model_id == 4:
                df["label"] = df["label"].apply(bin_labels)

            true_labels = df["label"].values - add
            y_score = df[prob_cols].values

            classes = np.array(list(class_labels.keys()))
            
            y_true_bin = label_binarize(true_labels, classes=classes)

            fpr, tpr, roc_auc = {}, {}, {}

            for i, cls in enumerate(classes):
                y_true_cls = y_true_bin[:,i]
                fpr[cls], tpr[cls], _ = roc_curve(y_true_cls, y_score[:, i])
                roc_auc[cls] = auc(fpr[cls], tpr[cls])

            macro_auc = roc_auc_score(y_true_bin, y_score, average='macro')

            results = pd.DataFrame({
                "Class": [class_labels[c] for c in classes],
                "AUC": [roc_auc[c] for c in classes]
            })
            results.to_csv(csv_path+"/classAUC_RegionViT_"+str(model_id)+fusion_type+".csv", index=False)

            print('')
            print("Results for model "+str(model_id)+" with "+str(batch_size)+"-slices")
            print('')
            print(results.to_string(index=False))
            print(f"\nMacro-average AUC: {macro_auc:.4f}")

            plt.figure(figsize=(8,6))
            for cls in classes:
                label = f"{class_labels[cls]} (AUC = {roc_auc[cls]:.3f})"
                plt.plot(fpr[cls], tpr[cls], lw=2, label=label)

            plt.plot([0,1], [0,1], 'k--', lw=1)
            plt.xlabel("False Positive Rate",fontsize=16)
            plt.ylabel("True Positive Rate",fontsize=16)
            plt.title(f"One-vs-All ROC Curves (Macro AUC = {macro_auc:.3f})",fontsize=16)
            plt.legend(loc="lower right", fontsize=14)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(save_path+"/rocAUC_RegionViT_"+str(model_id)+fusion_type+".png", dpi=300)
            plt.show()
        
        else:
            print(f"File not found, skipping: {which_file}")
