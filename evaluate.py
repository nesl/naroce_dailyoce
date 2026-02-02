import argparse
import numpy as np
from pathlib import Path
import re
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, create_src_causal_mask, CEDataset, log_results, load_config
from train import test
from focal_loss import FocalLoss

from mamba_ssm.models.config_mamba import MambaConfig
from models import RNN, TCN, TSTransformer, BaselineMamba
from nar_model import NARMamba, AdapterMamba, NarocePipeline, StateNaroce


parser = argparse.ArgumentParser(description='Model Evaluation with Config File')
parser.add_argument('config', type=str, help='Path to config JSON file')
parser.add_argument('model', type=str, help='Model name')
parser.add_argument('sensor_dataset', type=int, help='Sensor training dataset size', choices=[2000, 4000, 6000, 8000, 10000])
parser.add_argument('eval_dataset', type=str, help='Test dataset duration', choices=['3min', '5min', '15min', '30min'])
parser.add_argument('seed', type=int, help='Random seed')
parser.add_argument('--nar_dataset', type=int, choices=[20000, 40000, 80000], required=False, help='NAR dataset size (required for naroce evaluation)')
parser.add_argument('--shift', type=float, default=0.0, help='Temporal shift factor (e.g., 0.1 for shifted test set)')
parser.add_argument('--fuse', type=float, default=0.0, help='Adjacent window fusion factor (e.g., 0.1 for fused test set)')

args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Path configuration
DATA_ROOT = config.get('data_root', './data/CE_dataset/standard')
LOG_ROOT = config.get('log_root', 'experiments/evaluate')
MODEL_ROOT = config.get('model_root', 'experiments/baseline/saved_model')

# Experiment type from config
experiment_type = config.get('experiment_type', 'baseline')  # 'baseline' or 'naroce'
is_baseline = (experiment_type == 'baseline')
is_naroce = (experiment_type == 'naroce')

# NAR dataset from command line (only used for naroce)
nar_dataset = args.nar_dataset
if is_naroce and nar_dataset is None:
    raise ValueError("--nar_dataset is required for naroce evaluation")

# Model architecture constants
NUM_CE_CLASSES = 10  # Complex event classes (multi-label binary outputs)
NUM_AE_CLASSES = 9   # Atomic event classes (NAR vocabulary size)

# Extract values from command line
model_name = args.model
sensor_dataset = args.sensor_dataset
eval_dataset = args.eval_dataset
seed = args.seed
shift = args.shift
fuse = args.fuse

set_seeds(seed)

# Extract config values
batch_size = config.get('batch_size', 256)
alpha = config.get('alpha', 0.8)

# Determine dataset type suffix (for file names)
dataset_suffix = ""
if shift != 0.0:
    dataset_suffix = f"_shift{shift}"
elif fuse != 0.0:
    dataset_suffix = f"_fuse{fuse}"

# Determine dataset type for logging (includes parameter value for grouping)
dataset_type = "standard"
if shift != 0.0:
    dataset_type = f"shifted_{shift}"
elif fuse != 0.0:
    dataset_type = f"fused_{fuse}"

# Setup logging (always include dataset_type for grouping)
if is_baseline:
    log_dir = f'{LOG_ROOT}/{dataset_type}/{model_name}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f'{log_dir}/{eval_dataset}-{model_name}-{sensor_dataset}-{seed}.log'
elif is_naroce:
    log_dir = f'{LOG_ROOT}/{dataset_type}/{model_name}/n_{nar_dataset}/s_{sensor_dataset}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f'{log_dir}/{eval_dataset}-{model_name}-{seed}.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w')]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
criterion = FocalLoss(gamma=2, alpha=alpha, task_type='multi-label')
has_state = False
if model_name == 'state_naroce_mamba2_12L':
    has_state = True

""" Load datasets """
if model_name == 'lstm' or model_name == 'tcn' or model_name == 'transformer' or model_name == 'mamba1' or model_name == 'mamba2' or is_naroce is True:
    test_data_file = f'{DATA_ROOT}/ce{eval_dataset}_test_data{dataset_suffix}.npy'
    test_label_file = f'{DATA_ROOT}/ce{eval_dataset}_test_labels{dataset_suffix}.npy'
elif model_name == 'ae_lstm' or model_name == 'ae_tcn' or model_name == 'ae_transformer' or model_name == 'ae_mamba1' or model_name == 'ae_mamba2':
    test_data_file = f'{DATA_ROOT}/ae2ce{eval_dataset}_test_data{dataset_suffix}.npy'
    test_label_file = f'{DATA_ROOT}/ae2ce{eval_dataset}_test_labels{dataset_suffix}.npy'
else:
    raise Exception("Undefined models.")

logger.info(f"Loading test data from {test_data_file}")
logger.info(f"Dataset type: {dataset_type}")
if shift != 0.0:
    logger.info(f"Temporal shift: {shift} ({shift*100:.0f}% of window)")
if fuse != 0.0:
    logger.info(f"Adjacent window fusion: {fuse} ({fuse*100:.0f}% previous + {(1-fuse)*100:.0f}% current)")

test_data = np.load(test_data_file)
test_labels = np.load(test_label_file)
test_dataset = CEDataset(test_data, test_labels)

test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False
                            )

logger.info(f"Test data shape: {test_loader.dataset.data.shape}")


""" Load NN models """

input_dim = test_loader.dataset.data.shape[-1]
nar_vocab_size = NUM_AE_CLASSES
output_dim = NUM_CE_CLASSES

if is_baseline:
    if model_name == 'lstm':
        model = RNN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layer=5)

    elif model_name == 'tcn':
        # model = TCN(input_size=input_dim, output_size=output_dim, num_channels=[256,256,256,256], kernel_size=5)
        model = TCN(input_size=input_dim, output_size=output_dim, num_channels=[128,128,256,256,256,256,256,128,128], kernel_size=3)

        
    elif model_name == 'transformer':
        model = TSTransformer(input_dim=input_dim, output_dim=output_dim, num_head=4, num_layers=6, pos_encoding=True)

    elif model_name == 'mamba1':
        mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba1"})
        model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

    elif model_name == 'mamba2':
        mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
        model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

    else:
        raise Exception("Model is not defined.")
    # Path to load model
    model_path = f'{MODEL_ROOT}/{model_name}/{model_name}-{sensor_dataset}-{seed}.pt'
    # Path to save results to csv
    fname = 'results.csv'
    results_path = f"{log_dir}/{eval_dataset}-{model_name}-{fname}"
    
elif is_naroce:
    if model_name == 'state_naroce_mamba2_12L':
        nar_name= 'state_mamba2_v1'
        adapter_name = 'mamba2_12L'
        adapter_model = AdapterMamba(d_model=input_dim, n_layer=12)
        # mamba2_v1
        mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    else:
        match = re.match(r"naroce_([a-zA-Z]+\d*)_(\d+)L", model_name)
        if match:
            adapter_model_type = match.group(1)  # Capture the model type (e.g., 'mamba1', 'mlp')
            num_layers = int(match.group(2))  # Capture the number of layers
        else:
            raise ValueError(f"Invalid model name format: {model_name}")
        
        nar_name = 'mamba1_v1'
        nar_mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba1"})
        # nar_name = 'mamba1_v2'
        # nar_mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba1", "d_state": 64})

        adapter_name = adapter_model_type + '_' + str(num_layers) + 'L'
        if adapter_model_type == 'mamba1':
            adapter_mamba_config = MambaConfig(d_model=input_dim, n_layer=num_layers, ssm_cfg={"layer": "Mamba1"})
            # adapter_mamba_config = MambaConfig(d_model=input_dim, n_layer=num_layers, ssm_cfg={"layer": "Mamba1", "d_state": 64})
            adapter_model = AdapterMamba(adapter_mamba_config)
        else:
            raise Exception("Model is not defined.")

    nar_model = NARMamba(nar_mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim).nar
    model = NarocePipeline(
        frozen_nar=nar_model,
        adapter_model=adapter_model,
    )
    # Path to load model
    model_path = f'{MODEL_ROOT}/{nar_name}-{nar_dataset}-{adapter_name}-{sensor_dataset}-{seed}.pt'
    # Path to save results to csv
    fname = 'results.csv'
    results_path = f"{log_dir}/{eval_dataset}-{model_name}-{nar_dataset}-{fname}"
    

logger.info(f"Loading model from {model_path}")
checkpoint = torch.load(model_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

logger.info(f"Model loaded successfully: {model_name}")
model_summary = summary(model, verbose=0)
logger.info(f"\n{model_summary}")
nar_info = f", NAR train size: {nar_dataset}" if is_naroce else ""
logger.info(f"Evaluating - Model: {model_name}, Test dataset: {eval_dataset}, Sensor train size: {sensor_dataset}{nar_info}, Seed: {seed}, Batch size: {batch_size}")



""" Evaluation """
src_causal_mask = create_src_causal_mask(test_data.shape[1]) if model_name == 'transformer' or model_name == 'ae_transformer' or model_name == 'soft_ae_transformer' else None

result = test(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        src_mask=src_causal_mask,
        device=device
        )

log_results(results_path, result, seed, sensor_dataset)

