#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():

    parser = argparse.ArgumentParser(description="Process project slices and split datasets.")
    parser.add_argument("main_path", type=str, help="Base directory path", required=True)
    parser.add_argument("project_name", type=str, help="Project name", required=True)
    parser.add_argument("num_slices", type=int, nargs='+', help="Central and total slice counts (e.g., 9 20)")
    
    args = parser.parse_args()

    main_path = args.main_path
    project_name = args.project_name
    num_slices = args.num_slices
    
    file_path_dir = f"{main_path}/{project_name}-files"
    img_path = f"{main_path}/{project_name}-imgs"
    reading_path = f"{file_path_dir}/original_file.csv"

    if not os.path.exists(reading_path):
        print(f"Error: Could not find {reading_path}")
        return

    df = pd.read_csv(reading_path)
    df_train_traj, df_test_traj = train_test_split(df, random_state=42, test_size=0.2)

    train_traj_path = f"{file_path_dir}/df_train_traj.csv"
    test_traj_path = f"{file_path_dir}/df_test_traj.csv"

    df_train_traj.to_csv(train_traj_path, index=False)
    df_test_traj.to_csv(test_traj_path, index=False)

    print(f"Initial split complete. Training: {len(df_train_traj)}, Testing: {len(df_test_traj)}")

    for slices in num_slices:
        print(f"\nProcessing {slices} slices...")
        file_setting_train(main_path, project_name, train_traj_path, img_path, slices)
        file_setting_test(main_path, project_name, test_traj_path, img_path, slices)

    print("\nAll tasks finished successfully.")

def file_setting_train(main_path, project_name, file_path, img_path, slices):
    df = pd.read_csv(file_path)
    train_df, val_df = train_test_split(df, random_state=42, test_size=0.2)

    if slices == 9:
        slice_range = range(8, 17)
        suffix = "_reduced"
    elif slices == 20:
        slice_range = range(1, 21)
        suffix = ""
    else:
        print(f"Skipping: slices must be 9 or 20 (received {slices})")
        return

    out_name1 = f"{main_path}/{project_name}-files/training_{project_name}{suffix}.csv"
    out_name2 = f"{main_path}/{project_name}-files/validation_{project_name}{suffix}.csv"

    available_files = {f for _, _, files in os.walk(img_path) for f in files if f.endswith(".npy")}

    def expand_and_filter(df_split):
        expanded_rows = []
        for _, row in df_split.iterrows():
            sid = row["sid"]
            for s in slice_range:
                fname = f"{sid}{s}.npy"
                if fname in available_files:
                    new_row = row.copy()
                    new_row["File name"] = fname
                    expanded_rows.append(new_row)
        return pd.DataFrame(expanded_rows)
    
    out_df1 = expand_and_filter(train_df)
    out_df2 = expand_and_filter(val_df)
    
    for o_df in [out_df1, out_df2]:
        if not o_df.empty:
            cols = [o_df.columns[0], "File name"] + [c for c in o_df.columns if c not in [o_df.columns[0], "File name"]]
            o_df[cols].to_csv(out_name1 if o_df is out_df1 else out_name2, index=False)

def file_setting_test(main_path, project_name, file_path, img_path, slices):
    test_df = pd.read_csv(file_path)
    
    if slices == 9:
        slice_range = range(8, 17)
        out_name = f"{main_path}/{project_name}-files/testing_{project_name}_reduced.csv"
    elif slices == 20:
        slice_range = range(1, 21)
        out_name = f"{main_path}/{project_name}-files/testing_{project_name}.csv"
    else:
        return

    available_files = {f for _, _, files in os.walk(img_path) for f in files if f.endswith(".npy")}

    expanded_rows = []
    for _, row in test_df.iterrows():
        sid = row["sid"]
        for s in slice_range:
            fname = f"{sid}{s}.npy"
            if fname in available_files:
                new_row = row.copy()
                new_row["File name"] = fname
                expanded_rows.append(new_row)
    
    out_df = pd.DataFrame(expanded_rows)
    if not out_df.empty:
        cols = [out_df.columns[0], "File name"] + [c for c in out_df.columns if c not in [out_df.columns[0], "File name"]]
        out_df[cols].to_csv(out_name, index=False)

if __name__ == "__main__":
    main()