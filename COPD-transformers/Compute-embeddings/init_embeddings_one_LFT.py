# ··························································
# ············· QUERALT MARTÍN-SALADICH, 2025 ··············
# ··············· Brigham & Women's Hospital ···············
# ················· Harvard Medical School ·················
# ··························································

import numpy as np
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, default="COPDGene")

args = parser.parse_args()

project_name = args.project_name

def npy_to_one(model_id,slices,project_name,how,which,axis):
    if slices == 9:
        if axis == "coronal":
            raise ValueError(f"Can only be the original axial model")
        else:
            slices = slices
    
    elif slices == 20:
        if axis == "coronal":
            axis_suffix = "-cor"
        elif axis == "axial":
            axis_suffix = ""

    input_dir = f"{project_name}-features-{slices}{axis_suffix}"
    output_dir = input_dir

    train = f"{how}_train_labels_{which}.npy"
    test = f"{how}_test_labels_{which}.npy"

    trainlabels = np.load(os.path.join(input_dir,train),allow_pickle=True)  # shape: (num_patients, num_tasks)
    testlabels = np.load(os.path.join(input_dir,test),allow_pickle=True)  # shape: (num_patients, num_tasks)

    trainlabels = trainlabels[:, model_id-1]    
    print(trainlabels.shape,flush=True)

    output_name = f"{how}_train_labels_{which}_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, trainlabels)

    testlabels = testlabels[:, model_id-1]
    print(testlabels.shape,flush=True)

    output_name = f"{how}_test_labels_{which}_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, testlabels)

    return

modalities_map = {"E": [1], "T":  [2], "C":  [3], 
                  "ET": [1,2], "EC": [1,3], "CT": [2,3], 
                  "ETC": [1,2,3]}

for axis in ["coronal","axial"]:
    if axis == "coronal":
        num_slices_list = [20]
    elif axis == "axial":
        num_slices_list = [9,20]
    for slices in num_slices_list:
        for how in ["concat","mean","weighted","max","min","var","std","mix","pca"]:
            for which in ["E","T","C","ET","CT","EC","ETC"]:
                modality_values = modalities_map.get(which)
                for model_id in modality_values:
                    print(f"\n--- One-label extraction for {model_id} using {slices} slices for {how} and {which} ---\n")
                    npy_to_one(model_id,slices,project_name,how,which,axis)

print("One-label extraction finished for 9 and 20 slices using multiple embeddings and fusion methods")