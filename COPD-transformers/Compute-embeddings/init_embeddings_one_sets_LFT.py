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

def npy_to_one(model_id,slices,project_name,how,which,set_num):
    input_dir = f"{project_name}-features_sets-{slices}"
    output_dir = input_dir

    test = f"{how}_test_labels_{which}_set{set_num}.npy"

    testlabels = np.load(os.path.join(input_dir,test),allow_pickle=True)  # shape: (num_patients, num_tasks)
    testlabels = testlabels[:, model_id-1]
    print(testlabels.shape,flush=True)

    output_name = f"{how}_test_labels_{which}_set{set_num}_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, testlabels)
    return

modalities_map = {"E": [1], "T":  [2], "C":  [3], 
                  "ET": [1,2], "EC": [1,3], "CT": [2,3], 
                  "ETC": [1,2,3]}

for set_num in [1,2]:
    for slices in [9,20]:
        for how in ["concat","mean","weighted","max","min","var","std","mix","pca"]:
            for which in ["E","T","C","ET","CT","EC","ETC"]:
                modality_values = modalities_map.get(which)
                for model_id in modality_values:
                    print(f"\n--- One-label extraction for {model_id} using {slices} slices for {how} and {which} set {set_num} ---\n")
                    npy_to_one(model_id,slices,project_name,how,which,set_num)

print("One-label extraction finished for 9 and 20 slices using multiple embeddings and fusion methods")