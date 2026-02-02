from typing import Optional, Sequence, Dict, Any
import numpy as np
import random
import pandas as pd
import os
import logging
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# def stats(label, results_estimated):
#     # label = np.concatenate(label, 0)
#     # results_estimated = np.concatenate(results_estimated, 0)
#     label_estimated = np.argmax(results_estimated, 1)
#     f1 = f1_score(label, label_estimated, average='weighted')
#     acc = np.sum(label == label_estimated) / label.size
#     return acc, f1


class CEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)
    

class MultiTaskCEDataset(Dataset):
    def __init__(self, data, labels, in_states, out_states):
        self.data = data
        self.labels = labels
        self.in_states = in_states
        self.out_states = out_states

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        in_state = self.in_states[index]
        out_state = self.out_states[index]
        return data, label, in_state, out_state

    def __len__(self):
        return len(self.data)


def create_src_causal_mask(sz):
    src_mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(torch.bool)
    return src_mask


# def log_results(output_file, result, seed, dataset, threshold=0.5, test_type='cross-subject'):
#     """
#     Log evaluation results to a CSV file.

#     Args:
#         output_file: Path to the CSV file
#         result: Dictionary containing evaluation metrics
#         seed: Random seed used
#         dataset: Training dataset size
#         threshold: Classification threshold (default: 0.5)
#         test_type: Type of test set - 'cross-subject' or 'within-subject' (default: 'cross-subject')
#     """
#     # Prepare the result as a single row
#     row = {
#         "Test_Type": test_type,
#         "Seed": seed,
#         "Dataset_Size": dataset,
#         "Threshold": threshold,
#         "HammingAcc": result["HammingAcc"],
#         "F1_macro": result["F1_macro"],
#         "mAP_macro": result["mAP_macro"],
#         "Precision_macro": result["Precision_macro"],
#         "Recall_macro": result["Recall_macro"],
#     }
#     # Add per-class F1, Precision, Recall, mAP as individual columns
#     for i, (f1, precision, recall, ap) in enumerate(zip(
#         result["F1_per_class"],
#         result["Precision_per_class"],
#         result["Recall_per_class"],
#         result["AP_per_class"]
#     )):
#         row[f"F1_{i}"] = f1
#         row[f"Precision_{i}"] = precision
#         row[f"Recall_{i}"] = recall
#         row[f"AP_{i}"] = ap

#     # Convert the row to a DataFrame
#     new_row_df = pd.DataFrame([row])

#     if os.path.exists(output_file):
#         # Load existing results
#         existing_df = pd.read_csv(output_file)

#         # Find and replace rows with the same dataset_size, seed, and test_type
#         mask = (existing_df["Dataset_Size"] == dataset) & (existing_df["Seed"] == seed) & (existing_df["Test_Type"] == test_type)
#         if mask.any():
#             # Replace the existing row by aligning columns explicitly
#             for col in new_row_df.columns:
#                 existing_df.loc[mask, col] = new_row_df.iloc[0][col]
#         else:
#             # Append the new row if no match is found
#             existing_df = pd.concat([existing_df, new_row_df], ignore_index=True)
#     else:
#         # If the file doesn't exist, start a new DataFrame
#         existing_df = new_row_df

#     # Save the updated DataFrame back to the CSV file
#     existing_df.to_csv(output_file, index=False)

#     logger.info(f"Results saved to {output_file}")

def log_results(output_file, result, seed, dataset, threshold=0.5):
    # Prepare the result as a single row
    row = {
        "Seed": seed,
        "Dataset_Size": dataset,
        "Threshold": threshold,
        "HammingAcc": result["HammingAcc"],
        "F1_macro": result["F1_macro"],
        "mAP_macro": result["mAP_macro"],
        "Precision_macro": result["Precision_macro"],
        "Recall_macro": result["Recall_macro"],
    }
    # Add per-class F1, Precision, Recall, mAP as individual columns
    for i, (f1, precision, recall, ap) in enumerate(zip(
        result["F1_per_class"],
        result["Precision_per_class"],
        result["Recall_per_class"],
        result["AP_per_class"]
    )):
        row[f"F1_{i}"] = f1
        row[f"Precision_{i}"] = precision
        row[f"Recall_{i}"] = recall
        row[f"AP_{i}"] = ap

    # Convert the row to a DataFrame
    new_row_df = pd.DataFrame([row])

    if os.path.exists(output_file):
        # Load existing results
        existing_df = pd.read_csv(output_file)
        
        # Find and replace rows with the same dataset_size and seed
        mask = (existing_df["Dataset_Size"] == dataset) & (existing_df["Seed"] == seed)
        if mask.any():
            # Replace the existing row by aligning columns explicitly
            for col in new_row_df.columns:
                existing_df.loc[mask, col] = new_row_df.iloc[0][col]
        else:
            # Append the new row if no match is found
            existing_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        # If the file doesn't exist, start a new DataFrame
        existing_df = new_row_df

    # Save the updated DataFrame back to the CSV file
    existing_df.to_csv(output_file, index=False)

    logger.info(f"Results saved to {output_file}")


def log_loss(output_file, summary, seed):
    # Example: Number of epochs
    epochs = len(summary['val_loss'])

    # Output files for train and validation loss
    train_loss_file = f"{output_file}-train_loss.csv"
    val_loss_file = f"{output_file}-val_loss.csv"

    # Prepare data for the current seed
    train_loss_row = pd.DataFrame({
        "Seed": [seed],
        **{f"Epoch_{i+1}": [np.mean(summary['train_loss'][i])] for i in range(epochs)}
    })
    val_loss_row = pd.DataFrame({
        "Seed": [seed],
        **{f"Epoch_{i+1}": [np.mean(summary['val_loss'][i])] for i in range(epochs)}
    })

    # Function to update or append a row in a CSV
    def update_or_append_pandas(file_name, new_row):
        if os.path.exists(file_name):
            # Load existing data
            df = pd.read_csv(file_name)
            # Remove rows with the same seed
            df = df[df["Seed"] != new_row["Seed"].iloc[0]]
            # Append the new row
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create a new DataFrame
            df = new_row
        
        # Save the updated DataFrame back to the file
        df.to_csv(file_name, index=False)

    # Update or append train loss
    update_or_append_pandas(train_loss_file, train_loss_row)

    # Update or append validation loss
    update_or_append_pandas(val_loss_file, val_loss_row)

    logger.info(f"Train loss updated in {train_loss_file}")
    logger.info(f"Validation loss updated in {val_loss_file}")


# ============================================================================
# Configuration utilities
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load all configuration from a JSON file.

    Args:
        config_path: Path to the JSON config file

    Returns:
        Dictionary containing all configuration parameters

    Example config structure for baseline:
        {
            "data_root": "./data/CE_dataset",
            "log_root": "./experiments/baseline/saved_logs",
            "model_root": "./experiments/baseline/saved_model",
            "model": "mamba1",
            "datasets": [2000, 4000, 6000, 8000, 10000],
            "seeds": [0, 17, 1243, 3674, 7341, 53, 97, 103, 191, 99719],
            "batch_size": 256,
            "learning_rate": 0.001,
            "n_epochs": 5000,
            "alpha": 0.8
        }

    Example config structure for naroce:
        {
            "data_root": "./data/CE_dataset",
            "log_root": "./experiments/naroce/saved_logs",
            "model_root": "./experiments/naroce/saved_model",
            "nar_model": "mamba1_v1",
            "adapter_model": "mamba1_12L",
            "train": "pipeline",
            "nar_datasets": [20000],
            "sensor_datasets": [2000],
            "seeds": [53, 97, 103, 191, 99719],
            "batch_size_nar": 256,
            "batch_size_adapter": 256,
            "learning_rate_nar": 0.0005,
            "learning_rate_adapter": 0.005,
            "n_epochs_nar": 5000,
            "n_epochs_adapter": 5000,
            "alpha": 0.8,
            "gpu": 0
        }
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded config from {config_path}")
    return config