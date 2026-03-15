import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel
import os
import argparse
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    "axes.titlesize": 18,      # subplot titles
    "axes.labelsize": 16,      # x/y labels
    "xtick.labelsize": 16,     # x-tick labels
    "ytick.labelsize": 16,     # y-tick labels
    "legend.fontsize": 16,     # legend text
    "figure.titlesize": 20     # figure title
})
# -------------------------------------------------
# Arguments
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="main path")
parser.add_argument("--input", type=str, default="T")
parser.add_argument("--output", type=str, default="TRAJ")
parser.add_argument("--how", type=str, default="concat")
parser.add_argument("--add", type=int, default=0)
parser.add_argument("--project_name", type=str, default="COPDGene")
parser.add_argument("--slices", type=int, default=20)
parser.add_argument("--wrap", type=str, help="modular, concat, gatt, or gate",default="modular")
args = parser.parse_args()

main_path = args.path
output_type = args.output
input_type = args.input
vars_add = args.add
how = args.how
project_name = args.project_name
wrapping_mode = args.wrap

# -------------------------------------------------
# Paths
# -------------------------------------------------
origin_path = os.path.join(main_path, project_name + '-results/models')
save_path = os.path.join(main_path, project_name + '-results/figures')
for p in [origin_path, save_path]:
    os.makedirs(p, exist_ok=True)

# -------------------------------------------------
# Task definitions
# -------------------------------------------------
if output_type == "EMPH":
    tasks = {1: {"prob_cols": ["probs_class0","probs_class1","probs_class2","probs_class3"],
                 "labels": {0:'None',1:'Mild',2:'Moderate',3:'High'}}}
elif output_type == "TRAJ":
    tasks = {1: {"prob_cols": ["probs_class1","probs_class2","probs_class3",
                               "probs_class4","probs_class5","probs_class6"],
                 "labels": {0:'Traj 1',1:'Traj 2',2:'Traj 3',
                            3:'Traj 4',4:'Traj 5',5:'Traj 6'}}}
elif output_type == "COPD":
    tasks = {1: {"prob_cols": ["probs_class-1","probs_class0","probs_class1",
                               "probs_class2","probs_class3","probs_class4"],
                 "labels": {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',
                            3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}}}

fusion_map = {20: 'W'}
raw_data = []

# -------------------------------------------------
# Load predictions & compute AUCs
# -------------------------------------------------
for phase in [1,2]:
    for model_id, task in tasks.items():
        prob_cols = task["prob_cols"]

        for num_slices, fusion_type in fusion_map.items():
            csv_file = os.path.join(
                origin_path,
                f"probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set{phase}.csv"
            )

            if not os.path.exists(csv_file):
                continue

            df = pd.read_csv(csv_file)
            y_true = df["label"].values
            y_score = df[prob_cols].values

            # ---- Macro AUC ----
            macro_auc = roc_auc_score(
                y_true, y_score, multi_class='ovr', average='macro'
            )

            raw_data.append({
                'Setting': f"{num_slices}-slices",
                'Phase': f"Phase {phase}",
                'Metric': 'Macro',
                'Class': 'Macro',
                'AUC': macro_auc
            })

            # ---- Per-class AUC ----
            for class_idx, class_name in task["labels"].items():
                y_bin = (y_true == class_idx).astype(int)
                class_auc = roc_auc_score(y_bin, y_score[:, class_idx])

                raw_data.append({
                    'Setting': f"{num_slices}-slices",
                    'Phase': f"Phase {phase}",
                    'Metric': 'Class',
                    'Class': class_name,
                    'AUC': class_auc
                })

# -------------------------------------------------
# DataFrames
# -------------------------------------------------
results_df = pd.DataFrame(raw_data)
macro_df = results_df[results_df["Metric"] == "Macro"]
class_df = results_df[results_df["Metric"] == "Class"]

# -------------------------------------------------
# Paired macro statistics
# -------------------------------------------------
pivot_macro = macro_df.pivot(index='Setting', columns='Phase', values='AUC')
phase_a, phase_b = pivot_macro.columns.tolist()

vals_a = pivot_macro[phase_a]
vals_b = pivot_macro[phase_b]

try:
    _, p_val = wilcoxon(vals_a, vals_b)
except ValueError:
    _, p_val = ttest_rel(vals_a, vals_b)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 6))

# ---- 1) Macro heatmap ----
plt.subplot(1, 3, 1)
sns.heatmap(
    pivot_macro,
    annot=True,
    cmap="YlGnBu",
    fmt=".3f"
)
plt.title(f"Macro AUC per Phase\nPaired p-value: {p_val:.4f}")

# ---- 2) Macro distribution ----
from sklearn.metrics import confusion_matrix

# ---- Load predictions for 20-slice setting ----
fusion_type = "W"  # 20 slices
phase_preds = {}
for phase in [1, 2]:
    csv_file = os.path.join(
        origin_path,
        f"probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set{phase}.csv"
    )
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        prob_cols = tasks[1]["prob_cols"]
        # Predicted label = argmax of probabilities
        phase_preds[phase] = df[prob_cols].values.argmax(axis=1)

# ---- Confusion matrix Phase1 vs Phase2 ----
cm = confusion_matrix(phase_preds[1], phase_preds[2])
cm_labels = [tasks[1]["labels"][i] for i in sorted(tasks[1]["labels"].keys())]

plt.subplot(1, 3, 2)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cm_labels,
    yticklabels=cm_labels
)
plt.xlabel("Phase 2 Predictions", fontsize=14)
plt.ylabel("Phase 1 Predictions", fontsize=14)
plt.title("Confusion Matrix: Phase1 vs Phase2 (20 slices)", fontsize=16)

# ---- 3) Per-class AUC ----
from sklearn.metrics import roc_curve, auc

plt.subplot(1, 3, 3)

classes = list(tasks[1]["labels"].values())
colors = sns.color_palette("tab10", len(classes))
fusion_type = "W"   # only 20-slice setting

for i, class_name in enumerate(classes):
    for phase, linestyle in zip([1, 2], ["-", "--"]):
        csv_file = os.path.join(
            origin_path,
            f"probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_set{phase}.csv"
        )
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file)
        y_true = df["label"].values
        prob_cols = tasks[1]["prob_cols"]
        y_score = df[prob_cols].values

        # One-vs-rest binary for current class
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            color=colors[i],
            linestyle=linestyle,
            linewidth=2,
            label=f"{class_name} Phase {phase} (AUC={roc_auc:.2f})"
        )

plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves per Class (Phase1 vs Phase2) - 20 Slices", fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(
        save_path,
        f"SET_LFT_{output_type}_{how}_{input_type}_{vars_add}_{wrapping_mode}_2PHASE.png"),
    dpi=300)
plt.close()

print("Finished 2-phase macro + per-class analysis")