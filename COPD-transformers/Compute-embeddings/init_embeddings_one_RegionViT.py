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

def npy_to_one(model_id,slices,project_name):
    input_dir = f"{project_name}{model_id}-features-{slices}"
    output_dir = input_dir

    train = "train_labels.npy"
    val = "valid_labels.npy"
    test = "test_labels.npy"

    trainlabels = np.load(os.path.join(input_dir,train))  # shape: (num_patients, num_tasks)
    testlabels = np.load(os.path.join(input_dir,test))  # shape: (num_patients, num_tasks)
    vallabels = np.load(os.path.join(input_dir,val))  # shape: (num_patients, num_tasks)

    trainlabels = trainlabels[:, model_id-1]
    print(trainlabels.shape,flush=True)
    output_name = f"train_labels_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, trainlabels)

    vallabels = vallabels[:, model_id-1]
    print(vallabels.shape,flush=True)
    output_name = f"valid_labels_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, vallabels)

    testlabels = testlabels[:, model_id-1]
    print(testlabels.shape,flush=True)
    output_name = f"test_labels_{model_id}.npy"
    out_path = os.path.join(output_dir, output_name)
    np.save(out_path, testlabels)

    return

for model_id in [1,2,3]:
    for slices in [9,20]:
        print(f"\nExtracting labels for model {model_id} using {slices} slices...")
        npy_to_one(model_id,slices,project_name)