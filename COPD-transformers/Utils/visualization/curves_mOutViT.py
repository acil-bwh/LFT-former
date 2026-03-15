# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2026 ··············
# ··············· Brigham & Women's Hospital ···············
# ·················· Harvard Medical School ················
# ··························································
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id")
parser.add_argument("--project_name", help="project prefix", default="COPDGene")
parser.add_argument("--input", type=str, help="input type", default="CT")
parser.add_argument("--how", type=str, help="fusion method", default="concat")
parser.add_argument("--output", nargs='+', help="diagnostic outputs", default=["E"])

args = parser.parse_args()

main_path = args.path
project_name = args.project_name
input_type = args.input
how = args.how
output_type = args.output
num_slices = 20  # Set based on your experimental setup

# ---------------- DICTIONARY LOADING ----------------
def load_dict_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_dict_by_keys(original_dict, keys_to_keep):
    return {k: v for k, v in original_dict.items() if k in keys_to_keep}

source_files = f"{main_path}/{project_name}-files"
loaded_dict_out = load_dict_from_json(f"{source_files}/dict_out_config_class.json")

available_outputs = [
    "emph_cat_P1", "finalgold_visit_P1", "distwalked_P1", "lung_density_vnb_P1",
    "FEV1pp_post_P1", "eosinphl_P2", "pctEmph_Thirona_P1", "Pi10_Thirona_P1",
    "ChangeFEV1pp_P1P2", "ChangeVA_LD_P1P2", "Changedistwalked_P1P2", "ChangeVNB_LD_P1P2"
]
calling_outputs = ["E", "C", "DW", "VNB-LD", "FEV1", "Eos", "pE", "Pi10", "dFEV1", "dVA-LD", "dDW", "dVNB-LD"]
output_mapping_dict = dict(zip(calling_outputs, available_outputs))

which_outputs = [output_mapping_dict[code] for code in output_type]
filtered_dict_out = filter_dict_by_keys(loaded_dict_out, which_outputs)

# ---------------- PATH SETUP ----------------
origin_path = os.path.join(main_path, f"{project_name}-results/models")
save_path = os.path.join(main_path, f"{project_name}-results/figures")
csv_path = os.path.join(main_path, f"{project_name}-results/metrics")

for p in [origin_path, save_path, csv_path]:
    os.makedirs(p, exist_ok=True)

fusion_map = {9: 'C', 20: 'W'}
fusion_type = fusion_map.get(num_slices, 'X')

# ---------------- ROC/AUC CALCULATION & PLOTTING ----------------
print("\n" + "="*60)
print(f"ROC/AUC EVALUATION - {project_name}")
print("="*60)

for task_key, task_labels in filtered_dict_out.items():
    
    if not isinstance(task_labels, list):
        print(f"[-] Skipping regression task: {task_key}")
        continue

    # Path to probability CSV from inference
    csv_file = os.path.join(origin_path, f"probs_mOutViT_{task_key}_{fusion_type}_{how}_{input_type}.csv")

    if not os.path.exists(csv_file):
        print(f"[-] File not found: {csv_file}")
        continue

    df = pd.read_csv(csv_file)
    y_true = df["label"].values.astype(int)
    prob_cols = [col for col in df.columns if col.startswith("probs_class")]
    y_score = df[prob_cols].values

    n_classes = len(task_labels)
    classes = np.arange(n_classes)
    
    # Binarize labels (Handle Binary vs Multiclass)
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    macro_auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')

    # --- Print AUC Per Label ---
    print(f"\nTASK: {task_key}")
    print(f"{'Clinical Label':<35} | {'AUC':<10}")
    print("-" * 50)
    for i in range(n_classes):
        print(f"{str(task_labels[i]):<35} | {roc_auc[i]:.4f}")
    print("-" * 50)
    print(f"{'MACRO-AVERAGE AUC':<35} | {macro_auc:.4f}")

    # --- Save Numeric Results ---
    results_df = pd.DataFrame({
        "Class_ID": classes,
        "Class_Name": [str(x) for x in task_labels],
        "AUC": [roc_auc[i] for i in range(n_classes)]
    })
    results_df.to_csv(os.path.join(csv_path, f"metrics_{task_key}_{fusion_type}_{how}_{input_type}.csv"), index=False)

    # --- ROC Plotting ---
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2, 
                 label=f"{task_labels[i]} (AUC={roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(f"ROC Performance: {task_key}\nMacro-AUC: {macro_auc:.3f}", fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    fig_name = f"rocAUC_{task_key}_{fusion_type}_{how}_{input_type}.png"
    plt.savefig(os.path.join(save_path, fig_name), dpi=300)
    plt.close()

print("\n" + "="*60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("="*60)