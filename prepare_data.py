#!/usr/bin/env python3
"""
Dataset Preparation Script for Human Activity Recognition (HAR)
--------------------------------------------------------------
Provides utilities to:
1. Generate a synthetic sample dataset for quick testing and local verification.
2. Download and extract the full official dataset from the UCI Machine Learning Repository.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil
import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "Phones_accelerometer.csv")


def generate_sample_dataset(output_path=CSV_PATH, samples_per_activity=10000):
    """Generate a realistic synthetic sample dataset matching UCI HHAR schema."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Generating synthetic sample dataset ({samples_per_activity} rows/class)...")

    activities = ["bike", "sit", "stairsdown", "stairsup", "stand", "walk"]
    users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    models = ["nexus4", "s3", "s3mini", "samsungold"]
    devices = ["nexus4_1", "nexus4_2", "s3_1", "s3_2", "s3mini_1", "s3mini_2", "samsungold_1", "samsungold_2"]

    np.random.seed(42)
    rows = []
    idx = 0
    base_time = 1424696633000000000

    # Activity accelerometer characteristics (means and std devs)
    activity_profiles = {
        "bike": ([-1.2, 7.8, 4.5], [2.0, 3.5, 3.0]),
        "sit": ([0.1, 9.8, 0.2], [0.3, 0.4, 0.3]),
        "stairsdown": ([0.5, 9.5, 2.0], [4.0, 5.0, 4.5]),
        "stairsup": ([0.2, 10.2, 1.8], [3.5, 4.5, 4.0]),
        "stand": ([0.0, 9.7, 0.1], [0.2, 0.3, 0.2]),
        "walk": ([0.3, 8.5, 3.0], [3.0, 4.0, 3.5]),
    }

    for act in activities:
        mean, std = activity_profiles[act]
        xs = np.random.normal(mean[0], std[0], samples_per_activity)
        ys = np.random.normal(mean[1], std[1], samples_per_activity)
        zs = np.random.normal(mean[2], std[2], samples_per_activity)
        act_users = np.random.choice(users, samples_per_activity)
        act_models = np.random.choice(models, samples_per_activity)
        act_devices = np.random.choice(devices, samples_per_activity)

        for i in range(samples_per_activity):
            idx += 1
            arr_time = base_time + idx * 50000000
            cre_time = arr_time - np.random.randint(1000, 100000)
            rows.append({
                "Index": idx,
                "Arrival_Time": arr_time,
                "Creation_Time": cre_time,
                "x": xs[i],
                "y": ys[i],
                "z": zs[i],
                "User": act_users[i],
                "Model": act_models[i],
                "Device": act_devices[i],
                "gt": act,
            })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset successfully saved to: {output_path}")
    print(f"Shape: {df.shape} | Activities: {df['gt'].value_counts().to_dict()}")


def download_uci_dataset(output_dir=DATA_DIR):
    """Download full dataset from UCI repository and extract Phones_accelerometer.csv."""
    os.makedirs(output_dir, exist_ok=True)
    target_csv = os.path.join(output_dir, "Phones_accelerometer.csv")

    if os.path.exists(target_csv):
        print(f"Found existing dataset at: {target_csv}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return

    zip_path = os.path.join(output_dir, "hhar_archive.zip")
    print(f"Downloading UCI Heterogeneity Activity Recognition Dataset (~780 MB)...")
    print(f"URL: {UCI_URL}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {mb:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(UCI_URL, zip_path, reporthook=report_progress)
        print("\nDownload complete! Extracting Phones_accelerometer.csv...")

        # Extract inner zip or csv
        with zipfile.ZipFile(zip_path, 'r') as outer_zip:
            # Look for inner zip
            inner_zip_name = None
            for name in outer_zip.namelist():
                if "activity recognition exp.zip" in name.lower():
                    inner_zip_name = name
                    break

            if inner_zip_name:
                print(f"Extracting nested archive '{inner_zip_name}'...")
                outer_zip.extract(inner_zip_name, output_dir)
                nested_path = os.path.join(output_dir, inner_zip_name)
                with zipfile.ZipFile(nested_path, 'r') as inner_zip:
                    for inner_name in inner_zip.namelist():
                        if inner_name.endswith("Phones_accelerometer.csv"):
                            with inner_zip.open(inner_name) as source, open(target_csv, "wb") as target:
                                shutil.copyfileobj(source, target)
                            print(f"Extracted: {target_csv}")
                            break
                # Cleanup nested zip
                if os.path.exists(nested_path):
                    os.remove(nested_path)
            else:
                # Direct check
                for name in outer_zip.namelist():
                    if name.endswith("Phones_accelerometer.csv"):
                        with outer_zip.open(name) as source, open(target_csv, "wb") as target:
                            shutil.copyfileobj(source, target)
                        print(f"Extracted: {target_csv}")
                        break

        # Cleanup outer zip
        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"Full dataset ready at: {target_csv}")
    except Exception as e:
        print(f"\nError during download/extraction: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for HAR project")
    parser.add_argument("--download", action="store_true", help="Download the full dataset from UCI (~780 MB)")
    parser.add_argument("--sample", action="store_true", help="Generate a synthetic sample dataset for instant testing")
    parser.add_argument("--samples-per-class", type=int, default=10000, help="Number of samples per activity for synthetic dataset (default: 10000)")
    parser.add_argument("--output", type=str, default=CSV_PATH, help=f"Path to output CSV (default: {CSV_PATH})")
    args = parser.parse_args()

    if args.download:
        download_uci_dataset(os.path.dirname(args.output))
    elif args.sample or not os.path.exists(args.output):
        generate_sample_dataset(args.output, args.samples_per_class)
    else:
        print(f"Dataset already exists at: {args.output}")
        print("Use --download to redownload the full UCI dataset or --sample to regenerate synthetic data.")


if __name__ == "__main__":
    main()
