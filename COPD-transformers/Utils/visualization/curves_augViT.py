import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")

args = parser.parse_args()

main_path = args.path

origin_path = main_path+'/COPDGene-results/models'
save_path = main_path+'/COPDGene-results/figures'
csv_path = main_path+'/COPDGene-results/metrics'

import os
for paths in [origin_path,save_path,csv_path]:
    if not os.path.exists(paths):
        os.makedirs(paths)

for output_id in [1,3,5]:
    if output_id == 1:
        prob_cols = ["probs0", "probs1", "probs2", "probs3"]
        class_labels = {0:'None',1:'Mild',2:'Moderate',3:'High'}
        add = 0
        model_id = 1
    elif output_id == 2:
        prob_cols = ["probs1", "probs2", "probs3", "probs4", "probs5", "probs6"]
        class_labels = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
        add = +1
        model_id = 3
    elif output_id == 3:
        prob_cols = ["probs-1", "probs0", "probs1", "probs2", "probs3", "probs4"]
        class_labels = {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}
        add = -1
        model_id = 3
    elif output_id == 5:
        prob_cols = ["probs0", "probs1", "probs2", "probs3", "probs4"]
        class_labels = {0:'No COPD 0',1:'GOLD 1',2:'GOLD 2',3:'GOLD 3',4:'GOLD 4'}
        model_id = 3
        add = 0

    for batch_size in [9,20]:
        if batch_size == 1:
            fusion_type = ''
        elif batch_size == 9:
            fusion_type = 'C'
        elif batch_size == 20:
            fusion_type = 'W'
        else:
            raise(Exception('Can only be 1, 9 or 20 slices'))   
    
        project_name = "COPDGene"
        which_file = os.path.join(origin_path, f"probs_augViT_{project_name}_i{model_id}-o{output_id}.csv")
        
        if os.path.exists(which_file):
            df = pd.read_csv(which_file)
            print(f"Loading: {which_file}")
            df = df.iloc[:,1:]

            true_labels = df["label"].values
            y_score = df[prob_cols].values

            classes = np.array(list(class_labels.keys()))
            
            y_true_bin = label_binarize(true_labels, classes=classes)

            fpr, tpr, roc_auc = {}, {}, {}

            for i, cls in enumerate(classes):
                y_true_cls = y_true_bin[:,i]
                fpr[cls], tpr[cls], _ = roc_curve(y_true_cls, y_score[:, i])
                roc_auc[cls] = auc(fpr[cls], tpr[cls])

            macro_auc = roc_auc_score(y_true_bin, y_score, average='weighted', multi_class='ovr')

            results = pd.DataFrame({
                "Class": [class_labels[c] for c in classes],
                "AUC": [roc_auc[c] for c in classes]})
            
            results.to_csv(csv_path+f"/classAUC_augViT_i{model_id}-o{output_id}.csv", index=False)

            print('')
            print("Results for augViT model "+str(model_id)+" with "+str(batch_size)+"-slices")
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
            plt.savefig(save_path+f"/rocAUC_augViT_i{model_id}-o{output_id}.png", dpi=300)
            plt.show()

        else:
            print(f"File not found, skipping: {which_file}")

output_id = 6
prob_cols = ["probs0", "probs1"]
class_labels = {0:'No COPD',1:'GOLD 1-4'}
model_id = 3

for batch_size in [9,20]:

    if batch_size == 1:
        fusion_type = ''
    elif batch_size == 9:
        fusion_type = 'C'
    elif batch_size == 20:
        fusion_type = 'W'
    else:
        raise Exception('Can only be 1, 9 or 20 slices')

    project_name = "COPDGene"

    which_file = os.path.join(
        origin_path,
        f"probs_augViT_{project_name}_i{model_id}-o{output_id}.csv")

    if os.path.exists(which_file):

        print(f"Loading: {which_file}")

        df = pd.read_csv(which_file).iloc[:,1:]

        true_labels = df["label"].values
        y_score = df["probs1"].values   # positive class probability

        fpr, tpr, _ = roc_curve(true_labels, y_score)
        auc_val = roc_auc_score(true_labels, y_score)

        print("\nResults for augViT model", model_id, "with", batch_size, "slices")
        print(f"AUC: {auc_val:.4f}")

        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")

        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.xlabel("False Positive Rate", fontsize=16)
        plt.ylabel("True Positive Rate", fontsize=16)
        plt.title("ROC Curve", fontsize=16)
        plt.legend(loc="lower right", fontsize=14)

        plt.tight_layout()

        plt.savefig(
            save_path + f"/rocAUC_augViT_i{model_id}-o{output_id}.png",
            dpi=300)

        plt.show()

    else:
        print(f"File not found, skipping: {which_file}")