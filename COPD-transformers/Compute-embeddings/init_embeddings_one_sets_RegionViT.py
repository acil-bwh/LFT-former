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

def npy_to_one(model_id,slices,project_name,set_num):
    input_dir = f"{project_name}{model_id}-features_sets-{slices}"
    output_dir = input_dir

    test = f"test_labels_set{set_num}.npy"

    testlabels = np.load(os.path.join(input_dir,test))  # shape: (num_patients, num_tasks)

    testlabels = testlabels[:, model_id-1]
    print(testlabels.shape,flush=True)
    output_name = f"test_labels_set{set_num}_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, testlabels)

    return

for set_num in [1,2]:
    for model_id in [1,2,3]:
        for slices in [9,20]:
            print(f"\nExtracting labels for model {model_id} using {slices} slices for set {set_num}...")
            npy_to_one(model_id,slices,project_name,set_num)