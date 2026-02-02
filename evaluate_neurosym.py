"""
Neurosymbolic Evaluation: AE Classifier + FSM
Evaluates the combination of neural AE classifier predictions with FSM-based CE detection.
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import json

from sklearn.metrics import average_precision_score

# Import FSMs from data directory
import sys
sys.path.append('./data')
from fsm import Event1FSM, Event2FSM, Event3FSM, Event4FSM, Event5FSM
from fsm import Event6FSM, Event7FSM, Event8FSM, Event9FSM, Event10FSM

parser = argparse.ArgumentParser(description='Neurosymbolic Evaluation (AE+FSM)')
parser.add_argument('data_root', type=str, help='Path to data root (e.g., ./data/CE_dataset/standard)')
parser.add_argument('output_dir', type=str, help='Output directory for logs and results')
parser.add_argument('eval_dataset', type=str, help='Test dataset duration', choices=['3min', '5min', '15min', '30min'])
parser.add_argument('--shift', type=float, default=0.0, help='Temporal shift factor (e.g., 0.1 for shifted test set)')
parser.add_argument('--fuse', type=float, default=0.0, help='Adjacent window fusion factor (e.g., 0.1 for fused test set)')
parser.add_argument('--soft', action='store_true', help='Use soft AE2CE predictions instead of one-hot')

args = parser.parse_args()

# Path configuration
DATA_ROOT = args.data_root
LOG_ROOT = args.output_dir

# Extract values from command line
eval_dataset = args.eval_dataset
shift = args.shift
fuse = args.fuse
use_soft = args.soft

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

# Setup logging directory
method = "softae_fsm" if use_soft else "ae_fsm"
log_dir = f'{LOG_ROOT}/{dataset_type}/{method}'
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = f'{log_dir}/{eval_dataset}-{method}.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w')]
)
logger = logging.getLogger(__name__)

# Load dataset
prefix = "softae2ce" if use_soft else "ae2ce"
test_data_file = f'{DATA_ROOT}/{prefix}{eval_dataset}_test_data{dataset_suffix}.npy'
test_label_file = f'{DATA_ROOT}/{prefix}{eval_dataset}_test_labels{dataset_suffix}.npy'

logger.info(f"Loading test data from {test_data_file}")
logger.info(f"Dataset type: {dataset_type}")
logger.info(f"Method: {method} (AE classifier + FSM)")
if shift != 0.0:
    logger.info(f"Temporal shift: {shift} ({shift*100:.0f}% of window)")
if fuse != 0.0:
    logger.info(f"Adjacent window fusion: {fuse} ({fuse*100:.0f}% previous + {(1-fuse)*100:.0f}% current)")

ae2ce_test_data = np.load(test_data_file)  # Shape: (n_samples, n_windows, 9)
ae2ce_test_labels = np.load(test_label_file)  # Shape: (n_samples, n_windows, 10)

logger.info(f"Test data shape: {ae2ce_test_data.shape}")
logger.info(f"Test labels shape: {ae2ce_test_labels.shape}")

# Load class list for AE mapping
config_file = './data/Multimodal/dataset_config_w2.5.json'
with open(config_file, 'r') as f:
    dataset_config = json.load(f)
class_list = dataset_config['classes']

# Detect window size from config file name
WINDOW_SIZE = 2.5  # Default from dataset_config_w2.5.json

logger.info(f"AE classes: {class_list}")
logger.info(f"Window size: {WINDOW_SIZE}s")

# ============================================================================
# Neural + FSM Evaluation
# ============================================================================

logger.info("\n" + "="*70)
logger.info("Starting Neural + FSM evaluation")
logger.info("="*70)

all_ce_labels_pred = []
all_ce_labels_true = []

for i in range(len(ae2ce_test_data)):
    # Get AE predictions (argmax of softmax probabilities)
    ae_preds = np.argmax(ae2ce_test_data[i], axis=-1)  # Shape: (n_windows,)

    # Initialize FSMs for this sample
    fsm_list = [
        Event1FSM(), Event2FSM(), Event3FSM(), Event4FSM(), Event5FSM(),
        Event6FSM(), Event7FSM(), Event8FSM(), Event9FSM(), Event10FSM()
    ]

    # Multi-label predictions: binary vector of shape (n_windows, 10)
    ce_preds_multilabel = np.zeros((len(ae_preds), 10), dtype=np.int8)

    # Process each window
    for j in range(len(ae_preds)):
        action_id = ae_preds[j]
        action = class_list[action_id]

        # Update all FSMs and collect active events
        for fsm_idx, fsm in enumerate(fsm_list):
            event_label = fsm.update_state(input=action)
            if event_label > 0:
                # Event is active, set corresponding bit
                ce_preds_multilabel[j, event_label - 1] = 1

    all_ce_labels_pred.append(ce_preds_multilabel)

    # Collect true labels (already multi-label binary vectors)
    all_ce_labels_true.append(ae2ce_test_labels[i])

# Stack predictions and true labels
all_ce_labels_pred = np.vstack(all_ce_labels_pred)  # Shape: (n_total_windows, 10)
all_ce_labels_true = np.vstack(all_ce_labels_true)  # Shape: (n_total_windows, 10)

logger.info(f"\nTotal windows evaluated: {len(all_ce_labels_pred)}")
logger.info(f"Predicted labels shape: {all_ce_labels_pred.shape}")
logger.info(f"True labels shape: {all_ce_labels_true.shape}")

# ============================================================================
# Compute Metrics (Multi-label, following train.py test() function)
# ============================================================================

logger.info("\n" + "="*70)
logger.info("Computing multi-label metrics")
logger.info("="*70)

# y_true and y_pred are already flattened to [N*L, 10]
y_true = all_ce_labels_true
y_pred = all_ce_labels_pred

# Hamming accuracy (element-wise accuracy)
hamming_acc = (y_pred == y_true).mean()

# Compute per-class metrics (following train.py implementation)
n_classes = 10
prec_per = np.zeros(n_classes)
rec_per = np.zeros(n_classes)
f1_per = np.zeros(n_classes)

for i in range(n_classes):
    y_true_i = y_true[:, i]
    y_pred_i = y_pred[:, i]

    tp = np.sum((y_pred_i == 1) & (y_true_i == 1))
    fp = np.sum((y_pred_i == 1) & (y_true_i == 0))
    fn = np.sum((y_pred_i == 0) & (y_true_i == 1))

    # Precision
    if tp + fp > 0:
        prec_per[i] = tp / (tp + fp)
    else:
        prec_per[i] = 0.0

    # Recall
    if tp + fn > 0:
        rec_per[i] = tp / (tp + fn)
    else:
        rec_per[i] = 0.0

    # F1
    if prec_per[i] + rec_per[i] > 0:
        f1_per[i] = 2 * prec_per[i] * rec_per[i] / (prec_per[i] + rec_per[i])
    else:
        f1_per[i] = 0.0

# Macro averages
prec_macro = prec_per.mean()
rec_macro = rec_per.mean()
f1_macro = f1_per.mean()

result = {
    "HammingAcc": float(hamming_acc),
    "F1_macro": float(f1_macro),
    "F1_per_class": f1_per,
    "Precision_macro": float(prec_macro),
    "Recall_macro": float(rec_macro),
    "Precision_per_class": prec_per,
    "Recall_per_class": rec_per,
}

# Logging
logger.info('Hamming Acc: {:.4f}, F1 Macro: {:.4f}, Precision Macro: {:.4f}, Recall Macro: {:.4f}'.format(
    result["HammingAcc"],
    result["F1_macro"],
    result["Precision_macro"],
    result["Recall_macro"]))

logger.info('Per-class F1: {}'.format(', '.join([f'{x:.4f}' for x in f1_per])))
logger.info('Per-class Precision: {}'.format(', '.join([f'{x:.4f}' for x in prec_per])))
logger.info('Per-class Recall: {}'.format(', '.join([f'{x:.4f}' for x in rec_per])))

# Event distribution
logger.info(f"\nPredicted Event Distribution:")
for event_idx in range(10):
    count = np.sum(y_pred[:, event_idx])
    logger.info(f"  Event {event_idx+1}: {count} windows active")

logger.info(f"\nTrue Event Distribution:")
for event_idx in range(10):
    count = np.sum(y_true[:, event_idx])
    logger.info(f"  Event {event_idx+1}: {count} windows active")

# Save results to CSV (using utils.log_results format, but without seed/dataset)
results_path = f"{log_dir}/{eval_dataset}-{method}-results.csv"

# Write header if file doesn't exist
if not Path(results_path).exists():
    with open(results_path, 'w') as f:
        f.write("HammingAcc,F1_macro,Precision_macro,Recall_macro,")
        f.write(",".join([f"F1_event{i+1}" for i in range(10)]) + "\n")

# Append results
with open(results_path, 'a') as f:
    f.write(f"{result['HammingAcc']:.4f},{result['F1_macro']:.4f},")
    f.write(f"{result['Precision_macro']:.4f},{result['Recall_macro']:.4f},")
    f.write(",".join([f"{x:.4f}" for x in result['F1_per_class']]) + "\n")

logger.info(f"\n✓ Results saved to {results_path}")
logger.info("="*70)
