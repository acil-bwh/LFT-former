# ············· QUERALT MARTÍN-SALADICH, 2024 ··············
# ···················· BWH + UPF + VHIR ····················
# ··························································
from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import warnings 
warnings.filterwarnings('ignore') 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
project_name = "COPDGene"

def Confusion_augViT(main_path,
    model_id,
    batch_size,
    output_id):
    
    if batch_size == 9:
        fusion_type = 'C'

    elif batch_size == 20:
        fusion_type = 'W'
    else:
        raise(Exception('Can only be 9 or 20 slices'))
    
    read_path = f"{main_path}{project_name}-results/models"
    save_path = f"{main_path}{project_name}-results/figures"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if model_id == 1:
        model_name = "Emphysema"
        print('···· EMPHYSEMA MODEL ····')
    elif model_id == 2:
        model_name = "Trajectories"
        print('···· TRAJECTORIES MODEL ····')
    elif model_id == 3:
        model_name = "COPD"
        print('···· COPD MODEL ····')
    else:
        raise Exception("Sorry, invalid model (must be 1...6)")

    if output_id == 1:
        classes = {0:'None',1:'Mild',2:'Moderate',3:'Extreme'}
        labels = [0,1,2,3]
        add = 0
    elif output_id == 2:
        classes = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
        labels = [1,2,3,4,5,6]
        add = -1
    elif output_id == 3:
        classes = {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}
        labels = [-1,0,1,2,3,4]
        add = 1
    elif output_id == 5:
        prob_cols = ["probs0", "probs1", "probs2", "probs3", "probs4"]
        labels =[0,1,2,3,4]
        classes = {0:'No COPD 0',1:'GOLD 1',2:'GOLD 2',3:'GOLD 3',4:'GOLD 4'}
        add = 1
    elif output_id == 6:
        prob_cols = ["probs0", "probs1"]
        classes = {0:'No COPD 0',1:'GOLD 1-4'}
        labels = [0,1]
        add = 1    
    else:
        raise Exception("Sorry, invalid output (must be 1...6)")

    plot_def = model_name+" augViT "+str(batch_size)+"-slices"

    path_list = read_path+f'/model_augViT_{project_name}_i{model_id}-o{output_id}.csv'
    csv3 = pd.read_csv(path_list, sep=",",header=0,low_memory=False)

    y_test = csv3.iloc[:,1].astype(int)
    print(np.unique(y_test))
    predictions = csv3.iloc[:,2].astype(int)
    print(np.unique(predictions))

    # ---- Cohen's kappa ----
    ck_standard = sklearn.metrics.cohen_kappa_score(y_test, predictions, weights=None)
    ck_linear = sklearn.metrics.cohen_kappa_score(y_test, predictions, weights='linear')
    ck_quadratic = sklearn.metrics.cohen_kappa_score(y_test, predictions, weights='quadratic')
    print(f"Standard Cohen's Kappa = {ck_standard:.3f}")
    print(f"Linear-weighted Cohen's Kappa = {ck_linear:.3f}")
    print(f"Quadratic-weighted Cohen's Kappa = {ck_quadratic:.3f}")

    # ---- Accuracy ----
    accuracy = (y_test == predictions).mean()
    print(f"Exact Accuracy = {accuracy*100:.2f}%")

    difference = np.abs(y_test - predictions)
    one_off_accuracy = (difference <= 1).mean()
    print(f"One-off Accuracy = {one_off_accuracy*100:.2f}%\n")

    print('Classification Report:')
    print(classification_report(y_test, predictions, digits=3))

    cf_matrix = confusion_matrix(y_test, predictions, labels=tuple(labels))

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
    plt.title(plot_def, fontsize=16)
    plt.xticks(rotation=45)  # Rotate labels 45 degrees
    plt.tight_layout()
    plt.savefig(save_path+f"/confusion_augViT_i{model_id}-o{output_id}.png", transparent=None, dpi=300, format=None,
            metadata=None, bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto', backend=None)
    
def OneOff_augViT(
        main_path,
        model_id,
        batch_size,
        output_id):
    
    if batch_size == 9:
        fusion_type = 'C'

    elif batch_size == 20:
        fusion_type = 'W'
    else:
        raise(Exception('Can only be 9 or 20 slices'))
    
    read_path = f"{main_path}{project_name}-results/models"
    save_path = f"{main_path}{project_name}-results/figures"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if model_id == 1:
        model_name = "Emphysema"
        print('···· EMPHYSEMA MODEL ····')
    elif model_id == 2:
        model_name = "Trajectories"
        print('···· TRAJECTORIES MODEL ····')
    elif model_id == 3:
        model_name = "COPD"
        print('···· COPD MODEL ····')
    else:
        raise Exception("Sorry, invalid model (must be 1...6)")

    if output_id == 1:
        classes = {0:'None',1:'Mild',2:'Moderate',3:'Extreme'}
        labels = [0,1,2,3]
        add = 0
    elif output_id == 2:
        classes = {0:'Traj 1',1:'Traj 2',2:'Traj 3',3:'Traj 4',4:'Traj 5',5:'Traj 6'}
        labels = [1,2,3,4,5,6]
        add = -1
    elif output_id == 3:
        classes = {0:'PRISm -1',1:'No COPD 0',2:'GOLD 1',3:'GOLD 2',4:'GOLD 3',5:'GOLD 4'}
        labels = [-1,0,1,2,3,4]
        add = 1
    elif output_id == 5:
        labels =[0,1,2,3,4]
        classes = {0:'No COPD 0',1:'GOLD 1',2:'GOLD 2',3:'GOLD 3',4:'GOLD 4'}
        add = 1
    elif output_id == 6:
        classes = {0:'No COPD 0',1:'GOLD 1-4'}
        labels = [0,1]
        add = 1    
    else:
        raise Exception("Sorry, invalid output (must be 1...6)")

    plot_def = model_name+" augViT "+str(batch_size)+"-slices"+" One-off"

    path_list = read_path+f'/model_augViT_{project_name}_i{model_id}-o{output_id}.csv'
    csv3 = pd.read_csv(path_list, sep=",",header=0,low_memory=False)

    y_test = csv3.iloc[:,1].astype(int)
    predictions = csv3.iloc[:,2].astype(int)
    
    difference = abs(y_test-predictions)
    predictions_off = np.zeros(len(difference))

    for i in range(len(difference)):
        v_i = difference[i]
        if v_i == 1:
            predictions_off[i] = y_test[i]
        else:
            predictions_off[i] = predictions[i]

    print('Accuracy one-off augViT: '+str(batch_size)+'-slice for '+model_name)
    print('')

    # ---- Cohen's kappa ----
    ck_standard = sklearn.metrics.cohen_kappa_score(y_test, predictions_off, weights=None)
    ck_linear = sklearn.metrics.cohen_kappa_score(y_test, predictions_off, weights='linear')
    ck_quadratic = sklearn.metrics.cohen_kappa_score(y_test, predictions_off, weights='quadratic')
    print(f"Standard Cohen's Kappa = {ck_standard:.3f}")
    print(f"Linear-weighted Cohen's Kappa = {ck_linear:.3f}")
    print(f"Quadratic-weighted Cohen's Kappa = {ck_quadratic:.3f}")

    # ---- Accuracy ----
    accuracy = (y_test == predictions_off).mean()
    print(f"Exact Accuracy = {accuracy*100:.2f}%")

    difference = np.abs(y_test - predictions_off)
    one_off_accuracy = (difference <= 1).mean()
    print(f"One-off Accuracy = {one_off_accuracy*100:.2f}%\n")

    print('Classification Report:')
    print(classification_report(y_test, predictions_off, digits=3))

    cf_matrix = confusion_matrix(y_test, predictions_off, labels=tuple(labels))
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
    plt.title(plot_def, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path+f"/confusion_oneoff_augViT_i{model_id}-o{output_id}.png", transparent=None, dpi=300, format=None,
            metadata=None, bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto', backend=None)

    return