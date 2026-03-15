# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2026 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_metric_profile_plots(metrics_folder, output_dir):
    """
    Creates one plot per Classifier.
    X-axis: Metrics (F1-Macro, Accuracy, etc.)
    Y-axis: Value
    Hue: Fusion Methods & Baselines
    """
    all_files = glob.glob(os.path.join(metrics_folder, "metrics_*.csv"))
    
    if not all_files:
        print(f"No CSV files found in {metrics_folder} matching 'metrics_*.csv'")
        return

    # Metrics we want to visualize on the X-axis
    target_metrics = ["F1-Macro", "Accuracy", "Weighted_Kappa", "One_Off_Acc"]
    
    compiled_rows = []
    baselines_processed = set()

    for file_path in all_files:
        # Extract fusion method name from filename (e.g., metrics_attention.csv -> ATTENTION)
        method_name = os.path.basename(file_path).replace("metrics_", "").replace(".csv", "").upper()
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            view = row['View']
            classifier = row['Classifier']
            
            # Labeling logic
            if view == 'Fused':
                method_label = f"Fused: {method_name}"
            else:
                method_label = f"Baseline: {view}"
                # Baselines are identical across files, only add once per classifier/view
                if (classifier, view) in baselines_processed:
                    continue
                baselines_processed.add((classifier, view))

            # Pivot metrics to long form for plotting
            for m in target_metrics:
                if m in row:
                    compiled_rows.append({
                        'Classifier': classifier,
                        'Method': method_label,
                        'Metric': m,
                        'Value': row[m]})

    final_df = pd.DataFrame(compiled_rows)
    os.makedirs(output_dir, exist_ok=True)

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    unique_classifiers = final_df['Classifier'].unique()

    for clf in unique_classifiers:
        plt.figure(figsize=(12, 7))
        clf_df = final_df[final_df['Classifier'] == clf]
        
        # Pointplot automatically connects points with the same 'Method' (hue) 
        # across the categorical 'Metric' (x-axis)
        ax = sns.lineplot(
            data=clf_df,
            x='Metric',
            y='Value',
            hue='Method',
            marker='o',          # Circular markers
            markersize=6,        # Clearly visible points
            linewidth=1.0,       # Thinner lines (default is usually ~2.0)
            alpha=0.8,           # Slight transparency for overlapping lines
            palette="tab10"      # Distinct colors
        )

        plt.title(f"Performance Profile: {clf}", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Metric Score", fontsize=11)
        plt.xlabel("Metric", fontsize=11)
        plt.ylim(0, 1.0)
        
        # Move legend to the right
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Strategies", fontsize='small')
        
        plt.tight_layout()
        
        clean_name = clf.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(output_dir, f"profile_thin_{clean_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved refined plot for {clf}: {save_path}")

folder = "COPDGene-results/PCA-2/metrics"
out = "COPDGene-results/PCA-2/plots"
build_metric_profile_plots(folder, out)