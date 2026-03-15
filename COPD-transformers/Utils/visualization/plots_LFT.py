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
def Confusion_LFT(main_path,
                    num_slices,
                    how,
                    project_name,
                    output_type,
                    input_type,
                    vars_add,
                    wrapping_mode,
                    axis):
    """
    Plots confusion matrices for the mOutViT model results saved as CSVs.
    - main_path: base directory
    - num_slices: 9 (fusion f) or 20 (fusion w)
    """

    #### ECLIPSE CONFIG
    v = "" #### "" for filtered or 0 for raw
    L = "v1" #### v1 for v1 labels or v2 for v2 labels

    if axis == "coronal":
        axis_suffix = "-cor"
    elif axis == "axial":
        axis_suffix = ""

    # ---------------- PATHS ----------------
    if num_slices == 9:
        fusion_type = 'C'
    elif num_slices == 20:
        fusion_type = 'W'
    else:
        raise Exception('Can only be 9 or 20 slices')

    read_path = os.path.join(main_path, f"{project_name}-results/models")
    save_path = os.path.join(main_path, f"{project_name}-results/figures")
    os.makedirs(save_path, exist_ok=True)

    # ---------------- TASK DEFINITION ----------------
    if output_type == "EMPH":
        model_name = "Emphysema"
        print('···· EMPHYSEMA MODEL ····')
        classes = {0:'None',1:'Mild',2:'Moderate',3:'High'}
        labels = [0,1,2,3]
        add = 0
    elif output_type == "TRAJ":
        model_name = "Trajectories"
        print('···· TRAJECTORIES MODEL ····')
        classes = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
        labels = [1,2,3,4,5,6]
        add = -1
    elif output_type == "COPD":
        model_name = "COPD"
        print('···· COPD MODEL ····')
        classes = {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}
        labels = [-1,0,1,2,3,4]
        add = 1
    elif output_type == "TRAP":
        model_name = "TRAP"
        print('···· TRAPPING MODEL ····')
        classes = {0:'Low',1:'Mild',2:'Moderate',3:'High'}
        labels = [0,1,2,3]
        add = 0
    else:
        raise Exception("Sorry, invalid model (must be 1, 2, 3 or 4)")

    # ---------------- READ CSV ----------------

    if project_name == "ECLIPSE":
        csv_path = os.path.join(
        read_path, f"model_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}_{L}_{v}.csv")
    else:
        csv_path = os.path.join(
        read_path, f"model_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}.csv")
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna()

    y_true = df["label"].astype(int)
    y_pred = df["predicted"].astype(int)
    y_true[y_true == -2] = 0

    print("Unique in real: ",np.unique(y_true))
    print("Unique in pred: ",np.unique(y_pred))

    print(f"\n···· {model_name.upper()} MODEL {axis_suffix} {how} {wrapping_mode} ({num_slices}-slices) ····")
    print(f"Loaded {len(y_true)} samples from {csv_path}")

    # ---------------- METRICS ----------------
    ck_standard = cohen_kappa_score(y_true, y_pred)
    ck_linear = cohen_kappa_score(y_true, y_pred, weights='linear')
    ck_quadratic = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    accuracy = (y_true == y_pred).mean()
    one_off_accuracy = (np.abs(y_true - y_pred) <= 1).mean()

    print(f"Cohen's Kappa (standard): {ck_standard:.3f}")
    print(f"Cohen's Kappa (linear):   {ck_linear:.3f}")
    print(f"Cohen's Kappa (quadratic):{ck_quadratic:.3f}")
    print(f"Exact Accuracy:           {accuracy * 100:.2f}%")
    print(f"One-off Accuracy:         {one_off_accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    # ---------------- CONFUSION MATRIX ----------------
    cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = cf_matrix.sum(axis=1, keepdims=True)
    cm_perc = cf_matrix / cm_sum.astype(float) * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    names = tuple(classes.values())
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=names)
    cmap = plt.get_cmap('Blues')
    disp.plot(ax=ax, colorbar=False, cmap=cmap)

    # annotate counts + %
    annot = np.empty_like(cf_matrix).astype(str)
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            c = cf_matrix[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f"{c}\n({p:.1f}%)"

    # remove default text
    for txt in ax.texts:
        txt.set_visible(False)

    from matplotlib import cm
    norm = cm.colors.Normalize(vmin=cf_matrix.min(), vmax=cf_matrix.max())
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            value = cf_matrix[i, j]
            color_intensity = cmap(norm(value))
            luminance = 0.299 * color_intensity[0] + 0.587 * color_intensity[1] + 0.114 * color_intensity[2]
            text_color = 'black' if luminance > 0.5 else 'white'
            ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=14, color=text_color)

    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    plt.title(f"{model_name} LFT {axis} {how} {wrapping_mode} ({num_slices}-slices)", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if project_name == "ECLIPSE":
        save_file = os.path.join(save_path, f"confusion_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}_{L}_{v}.png")
    else:
        save_file = os.path.join(save_path, f"confusion_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}.png")

    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix → {save_file}\n")

def OneOff_LFT(main_path,
                    num_slices,
                    how,
                    project_name,
                    output_type,
                    input_type,
                    vars_add,
                    wrapping_mode,
                    axis):
    """
    One-off Accuracy and confusion matrix plotting for mOutViT predictions.
    - task_name: 'taskEMPH', 'taskTRAJ', 'taskCOPD'
    - num_slices: 9 (fusion f) or 20 (fusion w)
    """

    #### ECLIPSE CONFIG
    v = "" #### "" for filtered or 0 for raw
    L = "v1" #### v1 for v1 labels or v2 for v2 labels

    # ---------------- PATHS ----------------
    if num_slices == 9:
        fusion_type = 'C'
    elif num_slices == 20:
        fusion_type = 'W'
    else:
        raise Exception('Can only be 9 or 20 slices')
    
    if axis == "coronal":
        axis_suffix = "-cor"
    elif axis == "axial":
        axis_suffix = ""

    read_path = os.path.join(main_path, f"{project_name}-results/models")
    save_path = os.path.join(main_path, f"{project_name}-results/figures")
    os.makedirs(save_path, exist_ok=True)

    # ---------------- TASK SETTINGS ----------------
    if output_type == "EMPH":
        model_name = "Emphysema"
        print('···· EMPHYSEMA MODEL ····')
        classes = {0:'None',1:'Mild',2:'Moderate',3:'High'}
        labels = [0,1,2,3]
        add = 0
    elif output_type == "TRAJ":
        model_name = "Trajectories"
        print('···· TRAJECTORIES MODEL ····')
        classes = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
        labels = [1,2,3,4,5,6]
        add = -1
    elif output_type == "COPD":
        model_name = "COPD"
        print('···· COPD MODEL ····')
        classes = {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}
        labels = [-1,0,1,2,3,4]
        add = 1
    elif output_type == "TRAP":
        model_name = "TRAP"
        print('···· TRAPPING MODEL ····')
        classes = {0:'Low',1:'Mild',2:'Moderate',3:'High'}
        labels = [0,1,2,3]
        add = 0
    else:
        raise Exception("Sorry, invalid model (must be 1, 2, 3 or 4)")
 
    # ---------------- READ CSV ----------------
    if project_name == "ECLIPSE":
        csv_path = os.path.join(
        read_path, f"model_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}_{L}_{v}.csv")
    else:
        csv_path = os.path.join(
        read_path, f"model_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    y_true = df["label"].astype(int)
    y_pred = df["predicted"].astype(int)
    y_true[y_true == -2] = 0

    print("Unique in real: ",np.unique(y_true))
    print("Unique in pred: ",np.unique(y_pred))
    
    # ---------------- ONE-OFF ADJUSTMENT ----------------
    difference = np.abs(y_true - y_pred)
    y_oneoff = np.where(difference == 1, y_true, y_pred)

    # ---------------- METRICS ----------------
    ck_standard = cohen_kappa_score(y_true, y_oneoff)
    ck_linear = cohen_kappa_score(y_true, y_oneoff, weights='linear')
    ck_quadratic = cohen_kappa_score(y_true, y_oneoff, weights='quadratic')
    accuracy = (y_true == y_oneoff).mean()
    one_off_accuracy = (np.abs(y_true - y_oneoff) <= 1).mean()

    print(f"\n···· {model_name.upper()} MODEL {axis} {how} {wrapping_mode} {input_type} for {output_type} task ({num_slices}-slices) One-off ····")
    print(f"Standard Cohen's Kappa = {ck_standard:.3f}")
    print(f"Linear-weighted Cohen's Kappa = {ck_linear:.3f}")
    print(f"Quadratic-weighted Cohen's Kappa = {ck_quadratic:.3f}")
    print(f"Exact Accuracy = {accuracy*100:.2f}%")
    print(f"One-off Accuracy = {one_off_accuracy*100:.2f}%\n")
    print(classification_report(y_true, y_oneoff, digits=3))

    # ---------------- CONFUSION MATRIX ----------------
    cf_matrix = confusion_matrix(y_true, y_oneoff, labels=labels)
    cm_sum = cf_matrix.sum(axis=1, keepdims=True)
    cm_perc = cf_matrix / cm_sum.astype(float) * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    names = tuple(classes.values())
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=names)
    cmap = plt.get_cmap('Blues')
    disp.plot(ax=ax, colorbar=False, cmap=cmap)

    # Annotate counts + %
    annot = np.empty_like(cf_matrix).astype(str)
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            c = cf_matrix[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f'{c}\n({p:.1f}%)'

    # Remove default text
    for txt in ax.texts:
        txt.set_visible(False)

    from matplotlib import cm
    norm = cm.colors.Normalize(vmin=cf_matrix.min(), vmax=cf_matrix.max())
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            value = cf_matrix[i, j]
            color_intensity = cmap(norm(value))
            luminance = 0.299 * color_intensity[0] + 0.587 * color_intensity[1] + 0.114 * color_intensity[2]
            text_color = 'black' if luminance > 0.5 else 'white'
            ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=14, color=text_color)

    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    plt.xticks(rotation=45)
    plt.title(f"{model_name} LFT {axis} {how} {wrapping_mode} ({num_slices}-slices) One-off", fontsize=16)
    plt.tight_layout()

    if project_name == "ECLIPSE":
        save_file = os.path.join(save_path, f"confusion_oneoff_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}_{L}_{v}.png")
    else:
        save_file = os.path.join(save_path, f"confusion_oneoff_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}{axis_suffix}.png")
    
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix {save_file}\n")