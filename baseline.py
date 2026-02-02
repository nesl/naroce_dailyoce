import argparse
import numpy as np
from pathlib import Path
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, create_src_causal_mask, CEDataset, log_results, log_loss, load_config
from train import train, test
from focal_loss import FocalLoss

from models import RNN, TCN, TSTransformer, BaselineMamba

from mamba_ssm.models.config_mamba import MambaConfig


parser = argparse.ArgumentParser(description='NN Model Evaluation with Config File')
parser.add_argument('config', type=str, help='Path to config JSON file')
parser.add_argument('model', type=str, choices=['lstm', 'tcn', 'transformer', 'ae_lstm', 'ae_tcn', 'ae_transformer',  \
                                                'mamba1', 'mamba2', 'ae_mamba1', 'ae_mamba2',\
                                                'soft_ae_lstm', 'soft_ae_tcn', 'soft_ae_transformer','soft_ae_mamba1'])
parser.add_argument('dataset', type=int, help='Dataset size', choices=[2000, 4000, 6000, 8000, 10000])
parser.add_argument('seed', type=int, help='Random seed')

args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Path configuration
DATA_ROOT = config.get('data_root', './data/CE_dataset/standard')
LOG_ROOT = config.get('log_root', './experiments/baseline/saved_logs')
MODEL_ROOT = config.get('model_root', './experiments/baseline/saved_model')

# Model architecture constants
NUM_CE_CLASSES = 10  # Complex event classes (multi-label: 10 event types)
NUM_AE_CLASSES = 9   # Atomic event classes

# Extract config values (command-line args override config file)
model_type = args.model  # From command line
dataset = args.dataset  # From command line
seed = args.seed  # From command line
batch_size = config.get('batch_size', 256)
n_epochs = config.get('n_epochs', 5000)
learning_rate = config.get('learning_rate', 1e-3)
alpha = config.get('alpha', 0.8)

# Determine iteration mode: fixed steps (idealized) vs fixed epochs (realistic)
if 'sample_iterations' in config:
    # Idealized windowing: fixed total training steps
    sample_iterations = config['sample_iterations']
    # Calculate epochs: sample_iterations = epochs * dataset
    n_epochs = int(sample_iterations / dataset)
    iter_mode = 'fixed_steps'
elif 'n_epochs' in config:
    # Realistic windowing: fixed number of epochs
    n_epochs = config['n_epochs']
    iter_mode = 'fixed_epochs'
else:
    raise ValueError("Config must specify either 'sample_iterations' or 'n_epochs'.")

set_seeds(seed)

# Setup logging
log_dir = f'{LOG_ROOT}/{model_type}_steps' if iter_mode == 'fixed_steps' else f'{LOG_ROOT}/{model_type}_epochs'
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = f'{log_dir}/{model_type}-{dataset}-{seed}.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w')]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Multilabel Focal Loss Parameter - alpha: {alpha}")
criterion = FocalLoss(gamma=2, alpha=alpha, task_type='multi-label')


""" Load datasets """

if model_type == 'lstm' or model_type == 'tcn' or model_type == 'transformer' or model_type == 'mamba1' or model_type == 'mamba2':
    train_data_file = f'{DATA_ROOT}/ce5min_train_data_{dataset}.npy'
    train_label_file = f'{DATA_ROOT}/ce5min_train_labels_{dataset}.npy'
    val_data_file = f'{DATA_ROOT}/ce5min_val_data.npy'
    val_label_file = f'{DATA_ROOT}/ce5min_val_labels.npy'
    test_data_file = f'{DATA_ROOT}/ce5min_test_data.npy'
    test_label_file = f'{DATA_ROOT}/ce5min_test_labels.npy'

elif model_type == 'ae_lstm' or model_type == 'ae_tcn' or model_type == 'ae_transformer' or model_type == 'ae_mamba1' or model_type == 'ae_mamba2':
    train_data_file = f'{DATA_ROOT}/ae2ce5min_train_data_{dataset}.npy'
    train_label_file = f'{DATA_ROOT}/ae2ce5min_train_labels_{dataset}.npy'
    val_data_file = f'{DATA_ROOT}/ae2ce5min_val_data.npy'
    val_label_file = f'{DATA_ROOT}/ae2ce5min_val_labels.npy'
    test_data_file = f'{DATA_ROOT}/ae2ce5min_test_data.npy'
    test_label_file = f'{DATA_ROOT}/ae2ce5min_test_labels.npy'

elif model_type == 'soft_ae_lstm' or model_type == 'soft_ae_tcn' or model_type == 'soft_ae_transformer' or model_type == 'soft_ae_mamba1':
    train_data_file = f'{DATA_ROOT}/softae2ce5min_train_data_{dataset}.npy'
    train_label_file = f'{DATA_ROOT}/softae2ce5min_train_labels_{dataset}.npy'
    val_data_file = f'{DATA_ROOT}/softae2ce5min_val_data.npy'
    val_label_file = f'{DATA_ROOT}/softae2ce5min_val_labels.npy'
    test_data_file = f'{DATA_ROOT}/softae2ce5min_test_data.npy'
    test_label_file = f'{DATA_ROOT}/softae2ce5min_test_labels.npy'

else:
    raise Exception("Unknown dataset.")

logger.info(f"Loading datasets from {train_data_file}")
ce_train_data = np.load(train_data_file)
ce_train_labels = np.load(train_label_file)
ce_val_data = np.load(val_data_file)
ce_val_labels = np.load(val_label_file)
ce_test_data = np.load(test_data_file)
ce_test_labels = np.load(test_label_file)

logger.info(f"Data shapes - Train: {ce_train_data.shape}, Val: {ce_val_data.shape}, Test: {ce_test_data.shape}")

ce_train_dataset = CEDataset(ce_train_data, ce_train_labels)
ce_train_loader = DataLoader(ce_train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             )
ce_val_dataset = CEDataset(ce_val_data, ce_val_labels)
ce_val_loader = DataLoader(ce_val_dataset,
                            batch_size=batch_size,
                            shuffle=False, 
                            )
ce_test_dataset = CEDataset(ce_test_data, ce_test_labels)
ce_test_loader = DataLoader(ce_test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False
                            )


""" Load NN models """

input_dim = ce_train_data.shape[-1]
output_dim = NUM_CE_CLASSES

if model_type == 'lstm':
    earlystop_min_delta = 1e-2
    model = RNN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layer=5)

elif model_type == 'tcn':
    earlystop_min_delta = 1e-2
    model = TCN(input_size=input_dim, output_size=output_dim, num_channels=[128,128,256,256,256,256,256,128,128], kernel_size=3)

elif model_type == 'transformer':
    earlystop_min_delta = 1e-1
    model = TSTransformer(input_dim=input_dim, output_dim=output_dim, num_head=4, num_layers=6, pos_encoding=True)

elif model_type == 'mamba1':
    earlystop_min_delta = 1e-2 # ? 1e-4?
    # mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba1", "d_state": 64})
    mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba1"}) 
    model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

elif model_type == 'mamba2':
    earlystop_min_delta = 1e-2
    mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

else:
    raise Exception("Model is not defined.")

model_summary = summary(model, verbose=0)
logger.info(f"\n{model_summary}")
logger.info(f"Baseline Model: {model_type}, Early stop min delta: {earlystop_min_delta}")
logger.info(f"Training config - Iter mode: {iter_mode}, Batch size: {batch_size}, Epochs: {n_epochs}, LR: {learning_rate}")


""" Training and Testing """

# Create directory if it doesn't exist
model_dir = f'{MODEL_ROOT}/{model_type}_steps/' if iter_mode == 'fixed_steps' else f'{MODEL_ROOT}/{model_type}_epochs/'
Path(f'{model_dir}').mkdir(parents=True, exist_ok=True)
model_path = f'{model_dir}/{model_type}-{dataset}-{seed}.pt'
fname = 'results.csv'
results_path = f"{log_dir}/{model_type}-{fname}"
loss_path = f"{log_dir}/{model_type}-{dataset}"


src_causal_mask = create_src_causal_mask(ce_train_data.shape[1]) if model_type == 'transformer' or model_type == 'ae_transformer' or model_type == 'soft_ae_transformer' else None

summary = train(
    model=model,
    train_loader=ce_train_loader,
    val_loader=ce_val_loader, 
    n_epochs=n_epochs,
    lr=learning_rate,
    criterion=criterion,
    min_delta=earlystop_min_delta,
    save_path=model_path,
    src_mask=src_causal_mask,
    device=device
    )

log_loss(loss_path, summary, args.seed)

# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])


result = test(
    model=model,
    data_loader=ce_test_loader,
    criterion=criterion,
    src_mask=src_causal_mask,
    device=device     
)

log_results(results_path, result, seed, dataset)