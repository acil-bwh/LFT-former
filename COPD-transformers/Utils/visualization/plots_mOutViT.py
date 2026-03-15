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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

# ---------------- SEEDING ----------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ---------------- CONFUSION PLOT FUNCTION ----------------
def Confusion_mOutViT(main_path,
                   num_slices,
                   dict_out,
                   task_dict,
                   how,
                   project_name,
                   input_type):
    
    # ---------------- PATHS ----------------
    fusion_type = 'C' if num_slices == 9 else 'W' if num_slices == 20 else 'X'
    read_path = os.path.join(main_path, f"{project_name}-results/models")
    save_path = os.path.join(main_path, f"{project_name}-results/figures")
    os.makedirs(save_path, exist_ok=True)

    for t_idx, (task_name, details) in enumerate(task_dict.items()):
        # ---------------- PREPARE LABELS ----------------
        # Get label list from dict_out (e.g., ["None", "Mild", "Severe"])
        task_labels_info = [str(name) for name in dict_out.get(task_name)]

        if not isinstance(task_labels_info, list):
            print(f"Skipping Confusion Matrix for regression task: {task_name}")
            continue

        # Create numeric labels [0, 1, 2...] and string classes
        labels = np.arange(len(task_labels_info))
        classes_map = {i: name for i, name in enumerate(task_labels_info)}
        
        # ---------------- READ CSV ----------------
        csv_path = os.path.join(read_path, f"model_mOutViT_{task_name}_{fusion_type}_{how}_{input_type}.csv")

        if not os.path.exists(csv_path):
            print(f"Results CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        # Apply the same adjustment used in inference/training
        adjustment = details.get("adjustment", 0)
        y_true = df["label"].astype(int) - adjustment
        y_pred = df["predicted"].astype(int) - adjustment

        print(f"\n···· {task_name} MODEL {how} of {input_type} ({num_slices}-slices) ····")

        # ---------------- METRICS ----------------
        # Filter metrics to ensure y_true/y_pred are within expected label range
        ck_standard = cohen_kappa_score(y_true, y_pred)
        ck_linear = cohen_kappa_score(y_true, y_pred, weights='linear')
        accuracy = (y_true == y_pred).mean()
        
        print(f"Cohen's Kappa (standard): {ck_standard:.3f}")
        print(f"Cohen's Kappa (linear):{ck_linear:.3f}")
        print(f"Exact Accuracy:           {accuracy * 100:.2f}%")

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=task_labels_info, digits=3))

        # ---------------- CONFUSION MATRIX ----------------
        cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Avoid division by zero for empty classes
        cm_sum = cf_matrix.sum(axis=1, keepdims=True)
        cm_sum[cm_sum == 0] = 1 
        cm_perc = cf_matrix / cm_sum.astype(float) * 100

        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=task_labels_info)
        cmap = plt.get_cmap('Blues')
        disp.plot(ax=ax, colorbar=False, cmap=cmap)

        # Annotate counts + % 
        # 
        for i in range(cf_matrix.shape[0]):
            for j in range(cf_matrix.shape[1]):
                val = cf_matrix[i, j]
                perc = cm_perc[i, j]
                
                # Determine text color based on background darkness
                color_val = cmap(val / cf_matrix.max())[:3]
                luminance = 0.299*color_val[0] + 0.587*color_val[1] + 0.114*color_val[2]
                text_col = "white" if luminance < 0.5 else "black"
                
                ax.text(j, i, f"{val}\n({perc:.1f}%)", ha='center', va='center', 
                        color=text_col, fontsize=12, fontweight='bold')

        # Clean default text
        for txt in ax.texts[:-len(cf_matrix)**2]: txt.set_visible(False)

        ax.set_xlabel('Predicted label', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)
        plt.title(f"{task_name} Confusion Matrix\n{how} ({num_slices} slices)", fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_file = os.path.join(save_path, f"confusion_{task_name}_{fusion_type}_{how}_{input_type}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()

def OneOff_mOutViT(main_path,
                   num_slices,
                   dict_out,  # Now passed to retrieve class names
                   task_dict,
                   how,
                   project_name,
                   input_type):
    
    # ---------------- PATHS ----------------
    fusion_type = 'C' if num_slices == 9 else 'W' if num_slices == 20 else 'X'
    read_path = os.path.join(main_path, f"{project_name}-results/models")
    save_path = os.path.join(main_path, f"{project_name}-results/figures")
    os.makedirs(save_path, exist_ok=True)

    for t_idx, (task_name, details) in enumerate(task_dict.items()):
        # ---------------- PREPARE LABELS ----------------
        task_labels_info = [str(name) for name in dict_out.get(task_name)]       

        # Skip if regression (One-off accuracy isn't applicable to continuous values)
        if not isinstance(task_labels_info, list):
            continue

        labels = np.arange(len(task_labels_info))
        
        # ---------------- READ CSV ----------------
        csv_path = os.path.join(read_path, f"model_mOutViT_{task_name}_{fusion_type}_{how}_{input_type}.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        adjustment = details.get("adjustment", 0)
        y_true = df["label"].astype(int) - adjustment
        y_pred = df["predicted"].astype(int) - adjustment

        # ---------------- ONE-OFF ADJUSTMENT ----------------
        # In medical grading, if the model is off by 1 class, we treat it as correct 
        # for this specific 'relaxed' visualization.
        difference = np.abs(y_true - y_pred)
        y_oneoff = np.where(difference == 1, y_true, y_pred)

        # ---------------- METRICS ----------------
        ck_standard = cohen_kappa_score(y_true, y_oneoff)
        accuracy = (y_true == y_oneoff).mean()

        print(f"\n···· {task_name} MODEL {how} {input_type} ({num_slices}-slices) ONE-OFF ····")
        print(f"Cohen's Kappa = {ck_standard:.3f}")
        print(f"Accuracy = {accuracy*100:.2f}%")
        print(classification_report(y_true, y_oneoff, target_names=task_labels_info, digits=3))

        # ---------------- CONFUSION MATRIX ----------------
        cf_matrix = confusion_matrix(y_true, y_oneoff, labels=labels)
        
        # Normalize for percentages
        cm_sum = cf_matrix.sum(axis=1, keepdims=True)
        cm_sum[cm_sum == 0] = 1 # avoid division by zero
        cm_perc = cf_matrix / cm_sum.astype(float) * 100

        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=task_labels_info)
        cmap = plt.get_cmap('Greens') # Changed to Greens to distinguish from standard
        disp.plot(ax=ax, colorbar=False, cmap=cmap)

        # Custom Annotations
        for i in range(cf_matrix.shape[0]):
            for j in range(cf_matrix.shape[1]):
                val = cf_matrix[i, j]
                perc = cm_perc[i, j]
                
                # Determine text color based on background luminance
                color_val = cmap(val / (cf_matrix.max() + 1e-6))[:3]
                luminance = 0.299*color_val[0] + 0.587*color_val[1] + 0.114*color_val[2]
                text_col = "white" if luminance < 0.5 else "black"
                
                ax.text(j, i, f"{val}\n({perc:.1f}%)", ha='center', va='center', 
                        color=text_col, fontsize=12, fontweight='bold')

        # Cleanup default mpl labels
        for txt in ax.texts[:-len(cf_matrix)**2]: txt.set_visible(False)

        ax.set_xlabel('One-off Adjusted Prediction', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)
        plt.xticks(rotation=45)
        plt.title(f"{task_name} One-off Accuracy\n{how} ({num_slices} slices)", fontsize=15)
        plt.tight_layout()

        save_file = os.path.join(save_path, f"confusion_oneoff_{task_name}_{fusion_type}_{how}_{input_type}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"Saved relaxed confusion matrix → {save_file}\n")