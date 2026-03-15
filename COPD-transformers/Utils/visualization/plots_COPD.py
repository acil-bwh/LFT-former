import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, cohen_kappa_score
import warnings 
warnings.filterwarnings('ignore') 

project_name = "COPDGene"

def Confusion_COPD(main_path, model_id, output_id, mode='original'):
    """
    mode: 'original' (-1..4), 'five_class' (0..4), or 'binary' (0..1)
    """
    read_path = f"{main_path}{project_name}-results/models"
    save_path = f"{main_path}{project_name}-results/figures"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Setup Labels and Class Names based on mode
    if output_id == 3:  # COPD Model
        if mode == 'original':
            classes = {0:'PRISm -1', 1:'No COPD', 2:'GOLD 1', 3:'GOLD 2', 4:'GOLD 3', 5:'GOLD 4'}
            labels = [0, 1, 2, 3, 4, 5]
            suffix = "_original"
        elif mode == 'five_class':
            classes = {1:'No COPD', 2:'GOLD 1', 3:'GOLD 2', 4:'GOLD 3', 5:'GOLD 4'}
            labels = [1, 2, 3, 4, 5]
            suffix = "_5class"
        elif mode == 'binary':
            classes = {0:'No COPD', 1:'COPD'}
            labels = [0, 1]
            suffix = "_binary"
    else:
        # Fallback for Emphysema/Trajectories if needed
        classes = {i: f"Class {i}" for i in range(6)}
        labels = list(range(6))
        suffix = f"_{mode}"

    # 2. Load Data
    # Adjust filename based on the mode we saved earlier
    file_name = f'/model_augViT_{project_name}_i{model_id}-o{output_id}{suffix}.csv'
    csv_path = read_path + file_name
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    y_true = df['label'].astype(int)
    y_pred = df['predicted'].astype(int)

    # 3. Calculate Metrics
    print(f'\n···· {mode.upper()} EVALUATION ····')
    print(f"Unique True: {np.unique(y_true)}")
    print(f"Unique Pred: {np.unique(y_pred)}")

    ck_standard = cohen_kappa_score(y_true, y_pred)
    accuracy = (y_true == y_pred).mean()
    
    print(f"Standard Cohen's Kappa = {ck_standard:.3f}")
    print(f"Accuracy = {accuracy*100:.2f}%")
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=3))

    # 4. Generate Confusion Matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=tuple(labels))

    cm_sum = cf_matrix.sum(axis=1, keepdims=True)
    cm_perc = cf_matrix / cm_sum.astype(float) * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    names = tuple(classes.values())
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=names)
    cmap = plt.get_cmap('Blues')
    disp.plot(ax=ax, colorbar=False, cmap=cmap)

    annot = np.empty_like(cf_matrix).astype(str)
    nrows, ncols = cf_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cf_matrix[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f'{c}\n({p:.1f}%)'  # ejemplo: 12 (34.5%)

    for txt in ax.texts:
        txt.set_visible(False)

    from matplotlib import cm
    norm = cm.colors.Normalize(vmin=cf_matrix.min(), vmax=cf_matrix.max())

    for i in range(nrows):
        for j in range(ncols):
            value = cf_matrix[i, j]
            color_intensity = cmap(norm(value))
            luminance = 0.299 * color_intensity[0] + 0.587 * color_intensity[1] + 0.114 * color_intensity[2]
            text_color = 'black' if luminance > 0.5 else 'white'
            ax.text(j, i, annot[i, j],
                    ha='center', va='center',
                    fontsize=14, color=text_color)
            
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    plt.xticks(rotation=45)  # Rotate labels 45 degrees
    plt.tight_layout()
    
    save_file = save_path + f"/confusion_augViT_i{model_id}-o{output_id}_{mode}.png"
    plt.savefig(save_file, dpi=300)
    print(f"Saved: {save_file}")