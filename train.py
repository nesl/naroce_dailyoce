from tqdm import tqdm
import copy
import numpy as np
import logging
import torch
from torch import nn

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, average_precision_score

import matplotlib.pyplot as plt

from nar_model import InferenceParams

# import wandb

logger = logging.getLogger(__name__)

def cross_validation_folds(dataset, k_folds):
    """
        Args:
            dataset: A Dataset class of training data to split into train & validation set
            k_folds: An integer representing the number of cross validation folds
        Returns:
            folds: A list of tuples representing (train, validation) indices
        """
    kf = KFold(n_splits=k_folds, shuffle=True)
    folds = []
    for train_idx, valid_idx in kf.split(dataset):
        folds.append((train_idx, valid_idx))
    return folds


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        logger.debug(f'Early-stop counter: {self.counter}')
        return False


@torch.no_grad()
def quick_metrics(logits, labels, tau=0.5):
    """
    Fast GPU-based metrics for multi-label classification.
    Args:
        logits: [N, L, C] model outputs (C=10 events)
        labels: [N, L, C] ground truth in {0,1}
        tau: threshold for binary prediction
    Returns:
        dict with hamming_acc and macro_f1
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= tau).int()
    labels = labels.int()

    # Hamming accuracy (element-wise match)
    hamming_acc = (preds == labels).float().mean()

    # TP/FP/FN per class (flatten spatial dimension, aggregate over batch*time)
    # Reshape to (N*L, C) for vectorized computation
    preds_flat = preds.reshape(-1, preds.size(-1))
    labels_flat = labels.reshape(-1, labels.size(-1))

    tp = (preds_flat & labels_flat).sum(dim=0).float()
    fp = (preds_flat & (1 - labels_flat)).sum(dim=0).float()
    fn = ((1 - preds_flat) & labels_flat).sum(dim=0).float()

    # Per-class precision/recall/F1
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1_per = 2 * prec * rec / (prec + rec + eps)
    macro_f1 = f1_per.mean()

    return {
        "hamming_acc": hamming_acc.item(),
        "macro_f1": macro_f1.item(),
    }


def train(model, train_loader, val_loader, n_epochs, lr, criterion, save_path, min_delta=1e-4, src_mask=None, device='cpu', best_val_f1=None):
    """
    Unified training function for multi-label CE classification.

    Args:
        best_val_f1: Optional initial best validation F1 score for finetuning.
                     If None, starts from -1.0 (fresh training).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.95], weight_decay=0.1)
    model.to(device)

    # Move criterion to device upfront
    criterion = criterion.to(device)

    early_stopper = EarlyStopper(patience=30, min_delta=min_delta)
    summary = {'train_loss': [[] for _ in range(n_epochs)],
               'val_loss': [[] for _ in range(n_epochs)]}

    # Initialize best_val_f1: use provided value for finetuning, or -1.0 for fresh training
    if best_val_f1 is None:
        best_val_f1 = -1.0
        logger.info("Starting fresh training (best_val_f1 initialized to -1.0)")
    else:
        logger.info(f"Continuing from previous best_val_f1: {best_val_f1:.4f}")

    for e in tqdm(range(n_epochs)):
        logger.debug(f"Epoch: {e}")
        model.train()

        train_logits_all = []
        train_labels_all = []
        for batch in train_loader:
            optimizer.zero_grad()
            data = batch[0].to(device)
            labels = batch[1].to(device).float()  # Multi-label: (N, L, 10)

            # Forward pass
            if src_mask is not None:  # For transformer
                src_mask = src_mask.to(device)
                x = model(data, src_mask)
            else:
                x = model(data)  # (N, L, 10)

            # Calculate loss - no transpose needed for multi-label BCE
            loss = criterion(x, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            summary['train_loss'][e].append(loss.item())

            # Collect for epoch-level metrics
            train_logits_all.append(x.detach())
            train_labels_all.append(labels)

        # Compute epoch-level training metrics
        train_logits_epoch = torch.cat(train_logits_all, dim=0)
        train_labels_epoch = torch.cat(train_labels_all, dim=0)
        train_metrics = quick_metrics(train_logits_epoch, train_labels_epoch)

        # Validation
        model.eval()
        val_logits_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                labels = batch[1].to(device).float()

                # Forward pass
                if src_mask is not None:
                    x = model(data, src_mask)
                else:
                    x = model(data)

                # Calculate validation loss
                loss = criterion(x, labels)
                summary['val_loss'][e].append(loss.item())

                # Collect for epoch-level metrics
                val_logits_all.append(x)
                val_labels_all.append(labels)

        # Compute epoch-level validation metrics
        val_logits_epoch = torch.cat(val_logits_all, dim=0)
        val_labels_epoch = torch.cat(val_labels_all, dim=0)
        val_metrics = quick_metrics(val_logits_epoch, val_labels_epoch)

        # Logging (using raw loss values without approximation)
        logger.info('Train Loss: {}, Train Acc: {:.4f}, Train F1: {:.4f} | Val Loss: {}, Val Acc: {:.4f}, Val F1: {:.4f}'.format(
            np.mean(summary['train_loss'][e]),
            train_metrics['hamming_acc'],
            train_metrics['macro_f1'],
            np.mean(summary['val_loss'][e]),
            val_metrics['hamming_acc'],
            val_metrics['macro_f1']))

        # Save best model based on highest validation F1
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            checkpoint = {
                'best_epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['macro_f1']
                }
            torch.save(checkpoint, save_path)
            logger.info(f"Best model saved at epoch: {e}")

        # Early stopping
        if early_stopper.early_stop(np.mean(summary['val_loss'][e])):
            logger.info(f"Early-stop at epoch: {e}")
            break

    # Load best model from disk
    # model.load_state_dict(torch.load(save_path))

    return summary

@torch.inference_mode()
def test(model, data_loader, criterion, src_mask=None, device='cpu', threshold=0.5):
    """
    Validation for multilabel (events only, C=10). Computes:
      - Hamming accuracy (label-wise acc)
      - Macro F1 (events only)
      - mAP (events only, threshold-free)
    """
    model.eval()
    model.to(device)
    criterion = criterion.to(device)

    if src_mask is not None:
        src_mask = src_mask.to(device)

    loss_list = []

    all_labels = []
    all_preds = []   # thresholded 0/1
    all_probs = []   # sigmoid probs in [0,1] for mAP

    for batch in tqdm(data_loader):
        data = batch[0].to(device)       # [N,L,?]
        labels = batch[1].to(device).long()  # [N,L,10] in {0,1}

        # Forward
        if src_mask is not None:
            logits = model(data, src_mask)     # [N,L,10]
        else:
            logits = model(data)

        # Loss (with logits)
        loss = criterion(logits, labels.float())
        loss_list.append(loss.item())

        # Probs/preds for metrics
        probs = torch.sigmoid(logits)          # [N,L,10]
        preds = (probs >= threshold).long()

        # Collect without flattening - keep shape [N, L, 10]
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    # Stack - shape: [N_samples, seq_len, 10]
    y_true_seq = np.concatenate(all_labels, axis=0)
    y_pred_seq = np.concatenate(all_preds, axis=0)
    y_prob_seq = np.concatenate(all_probs, axis=0)

    # Log sample predictions in interpretable format (random samples, all timesteps)
    num_samples_to_log = min(20, len(y_pred_seq))  # Log 5 random samples
    random_sample_indices = np.random.choice(len(y_pred_seq), num_samples_to_log, replace=False)

    for sample_idx in random_sample_indices:
        logger.debug(f"=== Sample {sample_idx} ===")
        for t in range(y_pred_seq.shape[1]):
            pred_events = np.where(y_pred_seq[sample_idx, t] == 1)[0] + 1  # Convert 0-9 to 1-10
            true_events = np.where(y_true_seq[sample_idx, t] == 1)[0] + 1

            # If no events, it's class 0
            pred_classes = pred_events.tolist() if len(pred_events) > 0 else [0]
            true_classes = true_events.tolist() if len(true_events) > 0 else [0]

            # Mark mismatches
            match = np.array_equal(y_pred_seq[sample_idx, t], y_true_seq[sample_idx, t])
            error_mark = "" if match else " <= error!"
            logger.debug(f"  t={t}: Pred: {pred_classes}, True: {true_classes}{error_mark}")

    # Flatten for metrics: [N_samples * seq_len, 10]
    y_true = y_true_seq.reshape(-1, 10)
    y_pred = y_pred_seq.reshape(-1, 10)
    y_prob = y_prob_seq.reshape(-1, 10)

    # Hamming accuracy (calculated once on all data)
    hamming_acc = (y_pred == y_true).mean()

    # Compute all metrics per-class to avoid slow np.unique on full 2D array
    # This is much faster for large datasets (15min/30min with millions of samples)
    n_classes = 10
    prec_per = np.zeros(n_classes)
    rec_per = np.zeros(n_classes)
    f1_per = np.zeros(n_classes)
    AP_per = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        y_prob_i = y_prob[:, i]

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

        # AP (only if there are positive samples)
        if y_true_i.sum() > 0:
            AP_per[i] = average_precision_score(y_true_i, y_prob_i)
        else:
            AP_per[i] = 0.0

    # Macro averages
    prec_macro = prec_per.mean()
    rec_macro = rec_per.mean()
    f1_macro = f1_per.mean()
    mAP_macro = AP_per.mean()


    result = {
        "Loss": float(np.mean(loss_list)),
        "HammingAcc": float(hamming_acc),
        "F1_macro": float(f1_macro),
        "F1_per_class": f1_per,           # array len=10
        "Precision_macro": float(prec_macro),
        "Recall_macro": float(rec_macro),
        "Precision_per_class": prec_per,  # array len=10
        "Recall_per_class": rec_per,      # array len=10
        "mAP_macro": float(mAP_macro),
        "AP_per_class": AP_per,           # array len=10
    }

    # Logging
    logger.info('Test Loss: {}, Hamming Acc: {:.4f}, F1 Macro: {:.4f}, mAP Macro: {:.4f}'.format(
        result["Loss"],
        result["HammingAcc"],
        result["F1_macro"],
        result["mAP_macro"]))

    logger.info('Per-class F1: {}'.format(', '.join([f'{x}' for x in f1_per])))
    logger.info('Per-class mAP: {}'.format(', '.join([f'{x}' for x in AP_per])))

    return result
