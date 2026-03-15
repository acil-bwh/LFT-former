import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
import os
import argparse
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore') 

plt.rcParams.update({
    "axes.titlesize": 18,      # subplot titles
    "axes.labelsize": 16,      # x/y labels
    "xtick.labelsize": 14,     # x-tick labels
    "ytick.labelsize": 14,     # y-tick labels
    "legend.fontsize": 14,     # legend text
    "figure.titlesize": 20     # figure title
})

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

fusion_map = {9: 'C', 20: 'W'}
kernel_performance_map = {}
raw_data_for_ranking = []
task_names = ["taskTRAJ"]

for kernel in [1,2,3,4,5,6]:
    kernel_scores = []
    for model_id, task in tasks.items():
        prob_cols = task["prob_cols"]
        for num_slices in [20]:
            fusion_type = fusion_map[num_slices]
            csv_file = os.path.join(origin_path, f"probs_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}_k{kernel}.csv")
            
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                y_true, y_score = df["label"].values, df[prob_cols].values
                auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                
                kernel_scores.append(auc)
                raw_data_for_ranking.append({
                    'Setting': f"{task_names[model_id-1]}_{fusion_type}",
                    'Kernel': f"K{kernel}",
                    'AUC': auc})

    kernel_performance_map[f"K{kernel}"] = kernel_scores

df_comp = pd.DataFrame(raw_data_for_ranking)
pivot_df = df_comp.pivot(index='Setting', columns='Kernel', values='AUC')

print("·"*60)
print(" DEEP KERNEL COMPARISON REPORT ")
print("·"*60)

from scipy.stats import friedmanchisquare
# Friedman Test (Global Significance)
stat, p_p = friedmanchisquare(*[pivot_df[k] for k in pivot_df.columns])
print(f"\nGlobal Statistical Significance (Friedman):")
print(f"   - p-value: {p_p:.6f} ({'Significant' if p_p < 0.05 else 'Not Significant'})")

# Mean Ranking (Who is consistently best?)
ranks = pivot_df.rank(axis=1, ascending=False)
mean_ranks = ranks.mean().sort_values()
print(f"\nMean Rankings (1.0 is perfect):")
for k, r in mean_ranks.items():
    print(f"   - {k}: {r:.2f}")

# Effect Size (Cohen's d between Best and Worst Kernel)
best_k, worst_k = mean_ranks.index[0], mean_ranks.index[-1]
best_vals, worst_vals = pivot_df[best_k], pivot_df[worst_k]
cohens_d = (np.mean(best_vals) - np.mean(worst_vals)) / np.sqrt((np.var(best_vals) + np.var(worst_vals)) / 2)
print(f"\nMaximum Effect Size (Best vs Worst):")
print(f"   - Cohen's d: {cohens_d:.3f} ({'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'} effect)")

print(f"\nPairwise Win-Rate (%) - Row beats Column:")
win_matrix = pd.DataFrame(index=pivot_df.columns, columns=pivot_df.columns)
for k1 in pivot_df.columns:
    for k2 in pivot_df.columns:
        win_rate = (pivot_df[k1] > pivot_df[k2]).mean() * 100
        win_matrix.loc[k1, k2] = f"{win_rate:.1f}%"
print(win_matrix)

if p_p < 0.05:
    print(f"\nNemenyi Pairwise p-values (Significant if < 0.05):")
    posthoc = sp.posthoc_nemenyi_friedman(df_comp, value_col='AUC', group_col='Kernel')
    print(posthoc.round(4))


### Visualization of results:
results_df = pd.DataFrame(raw_data_for_ranking)

pivot_stats = results_df.pivot_table(index='Setting', columns='Kernel', values='AUC')
f_stat, p_val = stats.friedmanchisquare(*[pivot_stats[k] for k in pivot_stats.columns])

fig = plt.figure(figsize=(20, 6))
sns.set_style("whitegrid")

plt.subplot(1, 3, 1)
avg_performance = results_df.pivot_table(index='Setting', columns='Kernel', values='AUC', aggfunc='mean')
sns.heatmap(avg_performance, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title(f"Mean AUC-ROC per Kernel/Task\n(Friedman p-value: {p_val:.4f})")

plt.subplot(1, 3, 2)
sns.boxplot(x='Kernel', y='AUC', data=results_df, palette="Set2")
sns.swarmplot(x='Kernel', y='AUC', data=results_df, color=".25", size=4)
plt.title("AUC Distribution across all Tasks")
plt.ylim(0.6, 0.8)

plt.subplot(1, 3, 3)
sns.pointplot(x='Kernel', y='AUC', hue='Setting', data=results_df, markers=["o", "s"], linestyles=["-", "--"])
plt.title("Interaction: Fusion Type vs. Reconstruction Kernel")

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_path, f"KER_LFT_{output_type}_{fusion_type}-{how}_{input_type}_{vars_add}_{wrapping_mode}.png"), dpi=300)
plt.close()
print("Finished plotting comparisons")