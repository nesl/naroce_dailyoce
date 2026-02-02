import argparse
import numpy as np
from pathlib import Path
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, CEDataset, log_results, log_loss, load_config
from train import train, test
from focal_loss import FocalLoss

from mamba_ssm.models.config_mamba import MambaConfig
from nar_model import NARMamba, AdapterMamba, NarocePipeline, AdapterMLP

import re

parser = argparse.ArgumentParser(description='NAROCE Pipeline Training with Config File (Debug Mode)')
parser.add_argument('config', type=str, help='Path to config JSON file')
parser.add_argument('nar_dataset', type=int, help='NAR dataset size', choices=[20000, 40000, 80000])
parser.add_argument('seed', type=int, help='Random seed')
parser.add_argument('--sensor_dataset', type=int, choices=[2000, 4000, 6000, 8000, 10000], required=False, help='Sensor dataset size (required for adapter/pipeline training)')
parser.add_argument('--adapter_model', type=str, choices=['mamba1_2L', 'mamba1_3L', 'mamba1_4L', 'mamba1_6L','mamba1_12L', 'mamba2_6L', 'mamba2_12L', 'mlp_4L'], required=False, help='Adapter model (required for adapter/pipeline training)')

args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Extract config values
nar_model = config['nar_model']
train_mode = config['train']
nar_dataset = args.nar_dataset
seed = args.seed
alpha = config.get('alpha', 0.8)

# Path configuration
LOG_ROOT = config.get('log_root', 'experiments/naroce/saved_logs')
MODEL_ROOT = config.get('model_root', 'experiments/naroce/saved_model')

# Model architecture constants
NUM_CE_CLASSES = 10  # Complex event classes (multi-label: 10 event types)
NUM_AE_CLASSES = 9   # Atomic event classes (NAR vocabulary size)

# Load configurations based on training mode
if train_mode == 'nar':
    # NAR training: needs NAR symbolic data
    NAR_DATA_ROOT = config.get('nar_data_root', './data/CE_dataset/nar')
    batch_size_nar = config.get('batch_size_nar', 256)
    n_epochs_nar = config.get('n_epochs_nar', 5000)
    learning_rate_nar = config.get('learning_rate_nar', 5e-4)

elif train_mode in ['adapter', 'finetune']:
    # Adapter/finetune training: needs sensor data and adapter configs
    SENSOR_DATA_ROOT = config.get('sensor_data_root', './data/CE_dataset/standard')
    sensor_dataset = args.sensor_dataset
    adapter_model = args.adapter_model
    if sensor_dataset is None:
        raise ValueError(f"--sensor_dataset is required for {train_mode} training")
    if adapter_model is None:
        raise ValueError(f"--adapter_model is required for {train_mode} training")
    batch_size_adapter = config.get('batch_size_adapter', 256)

    # Determine iteration mode: fixed steps (idealized) vs fixed epochs (realistic)
    if 'sensor_sample_iterations' in config:
        # Idealized windowing: fixed total training steps
        sensor_sample_iterations = config['sensor_sample_iterations']
        # Calculate epochs: sensor_sample_iterations = epochs * sensor_dataset
        n_epochs_sensor = int(sensor_sample_iterations / sensor_dataset)
        iter_mode = 'fixed_steps'
    elif 'n_epochs_sensor' in config:
        # Realistic windowing: fixed number of epochs
        n_epochs_sensor = config['n_epochs_sensor']
        iter_mode = 'fixed_epochs'
    else:
        raise ValueError("Config must specify either 'sensor_sample_iterations' or 'n_epochs_sensor'.")

    learning_rate_adapter = config.get('learning_rate_adapter', 5e-3)
    adapter_dropout = config.get('adapter_dropout', 0.0)

    # Finetune-specific: learning rate for joint NAR+adapter training
    if train_mode == 'finetune':
        learning_rate_finetune = config.get('learning_rate_finetune', 1e-3)  

else:
    raise ValueError(f"Invalid train mode: {train_mode}. Must be 'nar', 'adapter', or 'finetune'")

set_seeds(seed)

# Setup logging
log_dir = f'{LOG_ROOT}/{train_mode}'
Path(log_dir).mkdir(parents=True, exist_ok=True)

if train_mode in ['adapter', 'finetune']:
    log_file = f'{log_dir}/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}-{seed}.log'
else:  # train_mode == 'nar'
    log_file = f'{log_dir}/{nar_model}-{nar_dataset}-{seed}.log'

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

# Load NAR datasets only for separate NAR training
if train_mode == 'nar':
    nar_train_data_file = f'{NAR_DATA_ROOT}/ct_ce5min_train_data_{nar_dataset}.npy'
    nar_train_label_file = f'{NAR_DATA_ROOT}/ct_ce5min_train_labels_{nar_dataset}.npy'
    nar_val_data_file = f'{NAR_DATA_ROOT}/ct_ce5min_val_data.npy'
    nar_val_label_file = f'{NAR_DATA_ROOT}/ct_ce5min_val_labels.npy'
    nar_test_data_file = f'{NAR_DATA_ROOT}/ct_ce5min_test_data.npy'
    nar_test_label_file = f'{NAR_DATA_ROOT}/ct_ce5min_test_labels.npy'

    logger.info(f"Loading NAR datasets from {nar_train_data_file}")
    nar_train_data = np.load(nar_train_data_file)
    nar_train_labels = np.load(nar_train_label_file)
    nar_val_data = np.load(nar_val_data_file)
    nar_val_labels = np.load(nar_val_label_file)
    nar_test_data = np.load(nar_test_data_file)
    nar_test_labels = np.load(nar_test_label_file)

    logger.info(f"NAR data shapes - Train: {nar_train_data.shape}, Val: {nar_val_data.shape}, Test: {nar_test_data.shape}")

    nar_train_dataset = CEDataset(nar_train_data, nar_train_labels)
    nar_val_dataset = CEDataset(nar_val_data, nar_val_labels)
    nar_test_dataset = CEDataset(nar_test_data, nar_test_labels)

    nar_train_loader = DataLoader(nar_train_dataset,
                                 batch_size=batch_size_nar,
                                 shuffle=True,
                                 )
    nar_val_loader = DataLoader(nar_val_dataset,
                                batch_size=batch_size_nar,
                                shuffle=False,
                                )
    nar_test_loader = DataLoader(nar_test_dataset,
                                batch_size=batch_size_nar,
                                shuffle=False
                                )

# Load sensor datasets for adapter/finetune training
if train_mode in ['adapter', 'finetune']:
    sensor_train_data_file = f'{SENSOR_DATA_ROOT}/ce5min_train_data_{sensor_dataset}.npy'
    sensor_train_label_file = f'{SENSOR_DATA_ROOT}/ce5min_train_labels_{sensor_dataset}.npy'
    sensor_val_data_file = f'{SENSOR_DATA_ROOT}/ce5min_val_data.npy'
    sensor_val_label_file = f'{SENSOR_DATA_ROOT}/ce5min_val_labels.npy'
    sensor_test_data_file = f'{SENSOR_DATA_ROOT}/ce5min_test_data.npy'
    sensor_test_label_file = f'{SENSOR_DATA_ROOT}/ce5min_test_labels.npy'

    logger.info(f"Loading sensor datasets from {sensor_train_data_file}")
    sensor_train_data = np.load(sensor_train_data_file)
    sensor_train_labels = np.load(sensor_train_label_file)
    sensor_val_data = np.load(sensor_val_data_file)
    sensor_val_labels = np.load(sensor_val_label_file)
    sensor_test_data = np.load(sensor_test_data_file)
    sensor_test_labels = np.load(sensor_test_label_file)

    logger.info(f"Sensor data shapes - Train: {sensor_train_data.shape}, Val: {sensor_val_data.shape}, Test: {sensor_test_data.shape}")

    sensor_train_dataset = CEDataset(sensor_train_data, sensor_train_labels)
    sensor_val_dataset = CEDataset(sensor_val_data, sensor_val_labels)
    sensor_test_dataset = CEDataset(sensor_test_data, sensor_test_labels)

    sensor_train_loader = DataLoader(sensor_train_dataset,
                                    batch_size=batch_size_adapter,
                                    shuffle=True,
                                    )
    sensor_val_loader = DataLoader(sensor_val_dataset,
                                    batch_size=batch_size_adapter,
                                    shuffle=False
                                    )
    sensor_test_loader = DataLoader(sensor_test_dataset,
                                    batch_size=batch_size_adapter,
                                    shuffle=False
                                    )


""" Load NN models """

# embedding_input_dim is always 128 (fusion embeddings dimension)
embedding_input_dim = 128
nar_vocab_size = NUM_AE_CLASSES
output_dim = NUM_CE_CLASSES

if nar_model == 'mamba2_v1':
    mamba_config = MambaConfig(d_model=embedding_input_dim, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    embed_nar_model = NARMamba(mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim)

elif nar_model == 'mamba1_v1':
    mamba_config = MambaConfig(d_model=embedding_input_dim, n_layer=12, ssm_cfg={"layer": "Mamba1"})
    embed_nar_model = NARMamba(mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim)

elif nar_model == 'mamba1_v2':
    mamba_config = MambaConfig(d_model=embedding_input_dim, n_layer=12, ssm_cfg={"layer": "Mamba1", "d_state": 64})
    embed_nar_model = NARMamba(mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim)

else:
    raise Exception("Model is not defined.")

model_summary = summary(embed_nar_model, verbose=0)
logger.info(f"\n{model_summary}")

""" Training and Testing """

# Create directory if it doesn't exist
Path(f'{MODEL_ROOT}/nar/').mkdir(parents=True, exist_ok=True)
# embed_nar_model_path = 'naroce/saved_model/nar/{}-{}-{}.pt'.format(nar_model, nar_dataset, seed)
embed_nar_model_path = f'{MODEL_ROOT}/nar/{nar_model}-{nar_dataset}-{seed}.pt'
fname = 'results.csv'
embed_nar_results_path = f"{LOG_ROOT}/nar/{nar_model}-{fname}"


if train_mode == 'nar':
    # Training NAR only
    logger.info(f"NAR Model: {nar_model}, Early stop min delta: 1e-2")
    logger.info(f"NAR Training config - Batch size: {batch_size_nar}, Epochs: {n_epochs_nar}, LR: {learning_rate_nar}")

    train(
        model=embed_nar_model,
        train_loader=nar_train_loader,
        val_loader=nar_val_loader, # need to generate validation data
        n_epochs=n_epochs_nar,
        lr=learning_rate_nar,
        criterion=criterion,
        save_path=embed_nar_model_path,
        min_delta=1e-2, # training nar
        device=device
        )

    # Load best model from disk before testing
    checkpoint = torch.load(embed_nar_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        embed_nar_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        embed_nar_model.load_state_dict(checkpoint)

    result = test(
        model=embed_nar_model,
        data_loader=nar_test_loader,
        criterion=criterion,
        device=device
        )

    log_results(embed_nar_results_path, result, seed, nar_dataset)

elif train_mode == 'adapter':
    # Adapter training: load pretrained NAR (frozen) + train new adapter
    nar_model_seed = config.get('nar_model_seed', 191)  # Which pretrained NAR seed to use
    logger.info(f"Loading pretrained NAR - Model: {nar_model}, NAR dataset: {nar_dataset}, NAR model seed: {nar_model_seed}")

    pretrained_nar_path = f'{MODEL_ROOT}/nar/{nar_model}-{nar_dataset}-{nar_model_seed}.pt'
    checkpoint = torch.load(pretrained_nar_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        embed_nar_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        embed_nar_model.load_state_dict(checkpoint)
    logger.info(f"Loaded pretrained NAR from: {pretrained_nar_path}")

    # Freeze NAR parameters
    nar = embed_nar_model.nar
    for param in nar.parameters():
        param.requires_grad = False

    # Parse adapter model name
    match = re.match(r"([a-zA-Z]+\d*)_(\d+)L", adapter_model)
    if match:
        adapter_model_type = match.group(1)
        num_layers = int(match.group(2))
    else:
        raise ValueError(f"Invalid model name format: {adapter_model}")

    # Create adapter
    if adapter_model_type == 'mamba1':
        mamba_config = MambaConfig(d_model=embedding_input_dim, n_layer=num_layers, ssm_cfg={"layer": "Mamba1"})
        adapter = AdapterMamba(mamba_config, dropout=adapter_dropout)
    elif adapter_model_type == 'mlp':
        adapter = AdapterMLP(input_dim=embedding_input_dim, hidden_dim=256, n_layer=num_layers, dropout=adapter_dropout)
    else:
        raise Exception(f"Adapter model type not defined: {adapter_model_type}")

    # Create pipeline model
    naroce_model = NarocePipeline(
        frozen_nar=nar,
        adapter_model=adapter,
    )

    params = sum(p.numel() for p in naroce_model.parameters())
    frozen_params = sum(p.numel() for p in naroce_model.parameters() if p.requires_grad is False)
    model_summary = summary(naroce_model, verbose=0)
    logger.info(f"\n{model_summary}")
    logger.info(f"NAROCE Model: {nar_model} (frozen) + {adapter_model}, Total params: {params:,}, Trainable: {params - frozen_params:,}")
    logger.info(f"Adapter Training config - Iter mode: {iter_mode}, Batch size: {batch_size_adapter}, Epochs: {n_epochs_sensor}, LR: {learning_rate_adapter}, Dropout: {adapter_dropout}, Early stop min delta: 1e-2")

    Path(f'{MODEL_ROOT}/pipeline/').mkdir(parents=True, exist_ok=True)
    naroce_model_path = f'{MODEL_ROOT}/pipeline/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}-{seed}.pt'
    fname = 'results.csv'
    naroce_results_path = f"{log_dir}/{nar_model}-{nar_dataset}-{adapter_model}-{fname}"
    loss_path = f"{log_dir}/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}"


    summary = train(
        model=naroce_model,
        train_loader=sensor_train_loader,
        val_loader=sensor_val_loader, # need to generate validation data
        n_epochs=n_epochs_sensor,
        lr=learning_rate_adapter,
        criterion=criterion,
        save_path=naroce_model_path,
        min_delta=1e-2, # training adapter
        device=device
        )

    log_loss(loss_path, summary, seed)

    # Load best model from disk
    naroce_model.load_state_dict(torch.load(naroce_model_path, map_location=device)['model_state_dict'])


    result = test(
        model=naroce_model,
        data_loader=sensor_test_loader,
        criterion=criterion,
        device=device
        )
    log_results(naroce_results_path, result, seed, sensor_dataset)

elif train_mode == 'finetune':
    # Finetune training: load pretrained adapter model (trained with same seed), unfreeze NAR + adapter, and finetune both
    logger.info(f"Loading pretrained adapter model - NAR: {nar_model}-{nar_dataset}, Adapter: {adapter_model}-{sensor_dataset}, Seed: {seed}")

    # Load pretrained adapter model (using same seed as current training)
    pretrained_adapter_path = f'{MODEL_ROOT}/pipeline/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}-{seed}.pt'
    checkpoint = torch.load(pretrained_adapter_path, map_location=device)

    # Parse adapter model name and create architecture
    match = re.match(r"([a-zA-Z]+\d*)_(\d+)L", adapter_model)
    if match:
        adapter_model_type = match.group(1)
        num_layers = int(match.group(2))
    else:
        raise ValueError(f"Invalid model name format: {adapter_model}")

    # Create adapter
    if adapter_model_type == 'mamba1':
        mamba_config = MambaConfig(d_model=embedding_input_dim, n_layer=num_layers, ssm_cfg={"layer": "Mamba1"})
        adapter = AdapterMamba(mamba_config, dropout=adapter_dropout)
    elif adapter_model_type == 'mlp':
        adapter = AdapterMLP(input_dim=embedding_input_dim, hidden_dim=256, n_layer=num_layers, dropout=adapter_dropout)
    else:
        raise Exception(f"Adapter model type not defined: {adapter_model_type}")

    # Create NAR model (will be loaded from checkpoint)
    nar = embed_nar_model.nar

    # Create pipeline model and load pretrained weights
    naroce_model = NarocePipeline(
        frozen_nar=nar,  # Will be unfrozen later
        adapter_model=adapter,
    )
    naroce_model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded pretrained adapter model from: {pretrained_adapter_path}")

    # UNFREEZE both NAR and adapter for joint finetuning
    for param in naroce_model.parameters():
        param.requires_grad = True

    params = sum(p.numel() for p in naroce_model.parameters())
    trainable_params = sum(p.numel() for p in naroce_model.parameters() if p.requires_grad)
    model_summary = summary(naroce_model, verbose=0)
    logger.info(f"\n{model_summary}")
    logger.info(f"NAROCE Model (Finetune): {nar_model} (unfrozen) + {adapter_model} (unfrozen), Total params: {params:,}, All trainable: {trainable_params:,}")
    logger.info(f"Finetune Training config - Iter mode: {iter_mode}, Batch size: {batch_size_adapter}, Epochs: {n_epochs_sensor}, LR: {learning_rate_finetune}, Dropout: {adapter_dropout}, Early stop min delta: 1e-2")

    Path(f'{MODEL_ROOT}/finetune').mkdir(parents=True, exist_ok=True)
    naroce_model_path = f'{MODEL_ROOT}/finetune/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}-{seed}.pt'
    fname = 'results.csv'
    naroce_results_path = f"{log_dir}/{nar_model}-{nar_dataset}-{adapter_model}-{fname}"
    loss_path = f"{log_dir}/{nar_model}-{nar_dataset}-{adapter_model}-{sensor_dataset}"

    summary = train(
        model=naroce_model,
        train_loader=sensor_train_loader,
        val_loader=sensor_val_loader,
        n_epochs=n_epochs_sensor,
        lr=learning_rate_finetune,
        criterion=criterion,
        save_path=naroce_model_path,
        min_delta=1e-2,
        device=device
        )

    log_loss(loss_path, summary, seed)

    # Load best model from disk before testing
    naroce_model.load_state_dict(torch.load(naroce_model_path, map_location=device)['model_state_dict'])

    result = test(
        model=naroce_model,
        data_loader=sensor_test_loader,
        criterion=criterion,
        device=device
        )
    log_results(naroce_results_path, result, seed, sensor_dataset)
