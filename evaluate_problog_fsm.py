"""
ProbLog FSM Evaluation: Probabilistic FSM with Soft AE Predictions
Evaluates probabilistic finite state machines using soft AE classifier predictions.
"""

import argparse
import numpy as np
from pathlib import Path
import logging

from sklearn.metrics import average_precision_score

# Import ProbLog FSM infrastructure
import sys
sys.path.append('./pyFSM/src')
sys.path.append('./pyFSM/experiments')

from fsm_parser import parser
from ce_fsms import fsm_ce1, fsm_ce2, fsm_ce3, fsm_ce4, fsm_ce5
from ce_fsms import fsm_ce6, fsm_ce7, fsm_ce8, fsm_ce9, fsm_ce10


parser_args = argparse.ArgumentParser(description='ProbLog FSM Evaluation')
parser_args.add_argument('data_root', type=str, help='Path to data root (e.g., ./data/CE_dataset/standard)')
parser_args.add_argument('output_dir', type=str, help='Output directory for logs and results')
parser_args.add_argument('eval_dataset', type=str, help='Test dataset duration', choices=['3min', '5min', '15min', '30min'])
parser_args.add_argument('--shift', type=float, default=0.0, help='Temporal shift factor (e.g., 0.1 for shifted test set)')
parser_args.add_argument('--fuse', type=float, default=0.0, help='Adjacent window fusion factor (e.g., 0.1 for fused test set)')

args = parser_args.parse_args()

# Path configuration
DATA_ROOT = args.data_root
LOG_ROOT = args.output_dir

# Extract values from command line
eval_dataset = args.eval_dataset
shift = args.shift
fuse = args.fuse

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
method = "problog_fsm"
log_dir = f'{LOG_ROOT}/{dataset_type}/{method}'
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = f'{log_dir}/{eval_dataset}-{method}.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w')]
)
logger = logging.getLogger(__name__)

# Load dataset (only softae2ce)
test_data_file = f'{DATA_ROOT}/softae2ce{eval_dataset}_test_data{dataset_suffix}.npy'
test_label_file = f'{DATA_ROOT}/softae2ce{eval_dataset}_test_labels{dataset_suffix}.npy'

logger.info(f"Loading test data from {test_data_file}")
logger.info(f"Dataset type: {dataset_type}")
logger.info(f"Method: {method} (ProbLog FSM with soft AE predictions)")
if shift != 0.0:
    logger.info(f"Temporal shift: {shift} ({shift*100:.0f}% of window)")
if fuse != 0.0:
    logger.info(f"Adjacent window fusion: {fuse} ({fuse*100:.0f}% previous + {(1-fuse)*100:.0f}% current)")

softae2ce_test_data = np.load(test_data_file)  # Shape: (n_samples, n_windows, 9)
softae2ce_test_labels = np.load(test_label_file)  # Shape: (n_samples, n_windows, 10)

logger.info(f"Test data shape: {softae2ce_test_data.shape}")
logger.info(f"Test labels shape: {softae2ce_test_labels.shape}")

# ============================================================================
# Normalize softae2ce probabilities to sum to 1
# ============================================================================

logger.info("\n" + "="*70)
logger.info("Normalizing softae2ce probabilities")
logger.info("="*70)

# Check if normalization is needed
prob_sums = softae2ce_test_data.sum(axis=-1)
logger.info(f"Probability sums - min: {prob_sums.min():.6f}, max: {prob_sums.max():.6f}, mean: {prob_sums.mean():.6f}")

# Normalize to ensure sum=1 for each window
softae2ce_test_data_normalized = softae2ce_test_data / softae2ce_test_data.sum(axis=-1, keepdims=True)

logger.info(f"After normalization - all sums = 1.0: {np.allclose(softae2ce_test_data_normalized.sum(axis=-1), 1.0)}")

# ============================================================================
# Initialize ProbLog FSMs
# ============================================================================

logger.info("\n" + "="*70)
logger.info("Initializing ProbLog FSMs")
logger.info("="*70)

def init_fsm(fsm_description):
    """Initialize a ProbLog FSM from description string."""
    fsm = parser.parse(fsm_description)
    fsm.problog_ready()
    return fsm

fsm_descriptions = [fsm_ce1, fsm_ce2, fsm_ce3, fsm_ce4, fsm_ce5,
                    fsm_ce6, fsm_ce7, fsm_ce8, fsm_ce9, fsm_ce10]

logger.info(f"Number of FSMs: {len(fsm_descriptions)}")

# ============================================================================
# Helper functions
# ============================================================================

def softmax_to_dict(prob_vector):
    """Convert 9-dim probability vector to dict for ProbLog FSM."""
    return {
        'brush': prob_vector[0],
        'click': prob_vector[1],
        'drink': prob_vector[2],
        'eat': prob_vector[3],
        'flush': prob_vector[4],
        'sit': prob_vector[5],
        'type': prob_vector[6],
        'walk': prob_vector[7],
        'wash': prob_vector[8]
    }

def get_most_likely_state(probabilities_dict):
    """Get the state with highest probability."""
    return max(probabilities_dict, key=probabilities_dict.get)

# ============================================================================
# ProbLog FSM Evaluation
# ============================================================================

logger.info("\n" + "="*70)
logger.info("Starting ProbLog FSM evaluation")
logger.info("="*70)

all_ce_labels_pred = []
all_ce_labels_true = []

for i in range(len(softae2ce_test_data_normalized)):
    if i % 100 == 0:
        logger.info(f"Processing sample {i}/{len(softae2ce_test_data_normalized)}")

    # Initialize fresh FSMs for this sample
    fsm_list = [init_fsm(fsm_description) for fsm_description in fsm_descriptions]
    fsm_labels = list(range(1, len(fsm_list) + 1))  # [1, 2, ..., 10]

    # Process each window
    sample_preds = []
    for j in range(len(softae2ce_test_data_normalized[i])):
        # Get soft AE probabilities for this window
        ae_probs = softae2ce_test_data_normalized[i, j]
        input_dict = softmax_to_dict(ae_probs)

        # Check all FSMs for violations
        ce_label = 0
        for fsm_idx, fsm in enumerate(fsm_list):
            # Update FSM with probabilistic input
            violation = fsm.update_state(input_dict)

            # Get accept state probability
            accept_prob = violation.get('s_accept', 0.0)

            # Threshold is 1/num_states (uniform distribution threshold)
            threshold = 1.0 / len(fsm.states)

            # If accept state probability exceeds threshold, we have a violation
            if accept_prob > threshold:
                ce_label = fsm_labels[fsm_idx]
                fsm.reset()  # Reset FSM after violation detected
                break  # First violation wins (single-label)

        sample_preds.append(ce_label)

    all_ce_labels_pred.append(np.array(sample_preds))

    # Collect true labels (convert multi-label to single-label)
    # Take the first active event (or 0 if none)
    ce_labels_true = []
    for j in range(len(softae2ce_test_labels[i])):
        active_events = np.where(softae2ce_test_labels[i, j] == 1)[0]
        if len(active_events) > 0:
            ce_labels_true.append(active_events[0] + 1)  # Convert 0-9 to 1-10
        else:
            ce_labels_true.append(0)  # No event
    all_ce_labels_true.append(np.array(ce_labels_true))

# Flatten predictions and true labels
all_ce_labels_pred = np.concatenate(all_ce_labels_pred)  # Shape: (n_total_windows,)
all_ce_labels_true = np.concatenate(all_ce_labels_true)  # Shape: (n_total_windows,)

logger.info(f"\nTotal windows evaluated: {len(all_ce_labels_pred)}")
logger.info(f"Predicted labels shape: {all_ce_labels_pred.shape}")
logger.info(f"True labels shape: {all_ce_labels_true.shape}")

# ============================================================================
# Compute Metrics (Single-label classification)
# ============================================================================

logger.info("\n" + "="*70)
logger.info("Computing single-label metrics")
logger.info("="*70)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Overall metrics
accuracy = accuracy_score(all_ce_labels_true, all_ce_labels_pred)
f1_macro = f1_score(all_ce_labels_true, all_ce_labels_pred, average='macro')
f1_weighted = f1_score(all_ce_labels_true, all_ce_labels_pred, average='weighted')

# Positive class metrics (events 1-10, excluding class 0 which is "no event")
positive_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
f1_positive = f1_score(all_ce_labels_true, all_ce_labels_pred, labels=positive_labels, average='macro', zero_division=0)

# Per-class metrics
f1_per_class = f1_score(all_ce_labels_true, all_ce_labels_pred, average=None, zero_division=0)
precision_per_class = precision_score(all_ce_labels_true, all_ce_labels_pred, average=None, zero_division=0)
recall_per_class = recall_score(all_ce_labels_true, all_ce_labels_pred, average=None, zero_division=0)

precision_macro = precision_score(all_ce_labels_true, all_ce_labels_pred, average='macro', zero_division=0)
recall_macro = recall_score(all_ce_labels_true, all_ce_labels_pred, average='macro', zero_division=0)

result = {
    "Accuracy": float(accuracy),
    "F1_macro": float(f1_macro),
    "F1_weighted": float(f1_weighted),
    "F1_positive": float(f1_positive),
    "Precision_macro": float(precision_macro),
    "Recall_macro": float(recall_macro),
    "F1_per_class": f1_per_class,
    "Precision_per_class": precision_per_class,
    "Recall_per_class": recall_per_class,
}

# Logging
logger.info('Accuracy: {:.4f}, F1 Macro: {:.4f}, F1 Positive: {:.4f}, Precision Macro: {:.4f}, Recall Macro: {:.4f}'.format(
    result["Accuracy"],
    result["F1_macro"],
    result["F1_positive"],
    result["Precision_macro"],
    result["Recall_macro"]))

logger.info('Per-class F1: {}'.format(', '.join([f'{x:.4f}' for x in f1_per_class])))
logger.info('Per-class Precision: {}'.format(', '.join([f'{x:.4f}' for x in precision_per_class])))
logger.info('Per-class Recall: {}'.format(', '.join([f'{x:.4f}' for x in recall_per_class])))

# Event distribution
logger.info(f"\nPredicted Event Distribution:")
unique, counts = np.unique(all_ce_labels_pred, return_counts=True)
for label, count in zip(unique, counts):
    if label == 0:
        logger.info(f"  No Event: {count} windows")
    else:
        logger.info(f"  Event {int(label)}: {count} windows")

logger.info(f"\nTrue Event Distribution:")
unique, counts = np.unique(all_ce_labels_true, return_counts=True)
for label, count in zip(unique, counts):
    if label == 0:
        logger.info(f"  No Event: {count} windows")
    else:
        logger.info(f"  Event {int(label)}: {count} windows")

# Save results to CSV
results_path = f"{log_dir}/{eval_dataset}-{method}-results.csv"

# Write header if file doesn't exist
if not Path(results_path).exists():
    with open(results_path, 'w') as f:
        f.write("Accuracy,F1_macro,F1_positive,Precision_macro,Recall_macro,")
        f.write(",".join([f"F1_event{i}" for i in range(11)]) + "\n")  # 0-10 (11 classes)

# Append results
with open(results_path, 'a') as f:
    f.write(f"{result['Accuracy']:.4f},{result['F1_macro']:.4f},")
    f.write(f"{result['F1_positive']:.4f},{result['Precision_macro']:.4f},{result['Recall_macro']:.4f},")
    f.write(",".join([f"{x:.4f}" for x in result['F1_per_class']]) + "\n")

logger.info(f"\n✓ Results saved to {results_path}")
logger.info("="*70)
