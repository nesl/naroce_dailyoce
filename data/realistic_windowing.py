"""
Realistic Detection Window Post-Processing Utility

This script converts existing fixed-window (2.5s) CE datasets into realistic
variable-window datasets with majority-vote AE aggregation.

Supports two dataset types:
1. **Standard datasets** (--dataset-type standard): With embeddings
   - Subdivide AE labels → Aggregate via majority vote → Generate fresh embeddings → Compute CE labels
   - Requires raw sensor data and pretrained models (BEATs, LIMUBert, Fusion)
   - **AE2CE generation**: Use --ae2ce or --soft-ae2ce to convert existing realistic embeddings to AE2CE

2. **NAR datasets** (--dataset-type nar): Symbolic labels only
   - Subdivide AE labels → Aggregate via majority vote → Compute CE labels
   - Much faster, no embeddings or sensor data needed

Usage:
    # Standard dataset - Train split (input-dir contains index files)
    python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split train --n-samples 2000 --gpu-id 0

    # Standard dataset - Val split
    python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split val --gpu-id 0

    # Standard dataset - Test split (add --duration flag for 15min/30min)
    python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split test --duration 15 --gpu-id 0
    python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split test --duration 30 --gpu-id 0

    # NAR dataset - Train split (input-dir contains NAR data files)
    python realistic_windowing.py --dataset-type nar --input-dir CE_dataset/nar --detection-window 2.0 --output-dir CE_dataset/nar_w2.0s --split train --n-samples 40000

    # NAR dataset - Val split
    python realistic_windowing.py --dataset-type nar --input-dir CE_dataset/nar --detection-window 2.0 --output-dir CE_dataset/nar_w2.0s --split val

    # NAR dataset - Test split (add --duration flag for 15min/30min)
    python realistic_windowing.py --dataset-type nar --input-dir CE_dataset/nar --detection-window 2.0 --output-dir CE_dataset/nar_w2.0s --split test --duration 15
    python realistic_windowing.py --dataset-type nar --input-dir CE_dataset/nar --detection-window 2.0 --output-dir CE_dataset/nar_w2.0s --split test --duration 30

    # AE2CE generation from existing realistic windowing embeddings
    python realistic_windowing.py --dataset-type standard --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split train --n-samples 2000 --ae2ce
    python realistic_windowing.py --dataset-type standard --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split val --soft-ae2ce
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List
from collections import Counter
import sys
import torch
import json
from torch.utils.data import TensorDataset, DataLoader

# Import FSM classes for CE label re-computation
from fsm import (
    Event1FSM, Event2FSM, Event3FSM, Event4FSM, Event5FSM,
    Event6FSM, Event7FSM, Event8FSM, Event9FSM, Event10FSM
)

# Import embedding generation functions
from dataset_loader import (
    load_beats_model,
    load_limubert_model,
    load_multimodal_fusion_model,
    load_ae_classifier,
    generate_ae2ce_from_fusion
)


def subdivide_to_deltas(
    ae_labels: np.ndarray,
    window_sec: float = 2.5,
    delta_sec: float = 0.5
) -> np.ndarray:
    """
    Subdivide existing AE labels into fine-grained deltas by repeating.

    Args:
        ae_labels: AE labels, shape (n_samples, n_windows) with values 0-8
        window_sec: Duration of each original window in seconds (default: 2.5)
        delta_sec: Duration of each delta in seconds (default: 0.5)

    Returns:
        ae_deltas: Fine-grained AE labels, shape (n_samples, n_windows * deltas_per_window)
    """
    n_samples, n_windows = ae_labels.shape
    deltas_per_window = int(window_sec / delta_sec)

    # Repeat each label deltas_per_window times
    ae_deltas = np.repeat(ae_labels, deltas_per_window, axis=1)

    print(f"Subdivided {n_windows} windows ({window_sec}s each) into {ae_deltas.shape[1]} deltas ({delta_sec}s each)")
    print(f"  Deltas per window: {deltas_per_window}")

    return ae_deltas


def aggregate_with_majority_vote(
    ae_deltas: np.ndarray,
    detection_window_sec: float,
    delta_sec: float = 0.5
) -> np.ndarray:
    """
    Aggregate fine-grained deltas into detection windows using majority voting.

    Args:
        ae_deltas: Fine-grained AE labels, shape (n_samples, n_deltas)
        detection_window_sec: Duration of each detection window in seconds
        delta_sec: Duration of each delta in seconds (default: 0.5)

    Returns:
        aggregated_ae: Aggregated AE labels, shape (n_samples, n_detection_windows)
    """
    n_samples, n_deltas = ae_deltas.shape
    deltas_per_detection_window = int(detection_window_sec / delta_sec)

    # Calculate number of detection windows
    n_detection_windows = n_deltas // deltas_per_detection_window

    # Truncate to fit evenly into detection windows
    ae_deltas_truncated = ae_deltas[:, :n_detection_windows * deltas_per_detection_window]

    # Reshape to (n_samples, n_detection_windows, deltas_per_detection_window)
    ae_deltas_reshaped = ae_deltas_truncated.reshape(n_samples, n_detection_windows, deltas_per_detection_window)

    # Apply majority voting
    aggregated_ae = np.zeros((n_samples, n_detection_windows), dtype=np.int64)

    for i in range(n_samples):
        for j in range(n_detection_windows):
            window_labels = ae_deltas_reshaped[i, j, :]
            # Majority vote using Counter.most_common
            vote_counts = Counter(window_labels)
            aggregated_ae[i, j] = vote_counts.most_common(1)[0][0]

    print(f"Aggregated {n_deltas} deltas ({delta_sec}s each) into {n_detection_windows} detection windows ({detection_window_sec}s each)")
    print(f"  Deltas per detection window: {deltas_per_detection_window}")

    return aggregated_ae


def map_detection_window_to_raw_indices(
    window_idx: int,
    detection_window_sec: float,
    sample_indices: np.ndarray,
    original_window_sec: float = 2.5
) -> List[Tuple[int, float, float]]:
    """
    Determine which 2.5s raw windows contribute to a W-second detection window.

    O(1) time complexity - directly calculates contributing windows using math.

    Args:
        window_idx: Detection window index (0, 1, 2, ...)
        detection_window_sec: Detection window size in seconds (e.g., 2.0)
        sample_indices: (n_windows,) - sequential indices to 2.5s raw windows
        original_window_sec: Original window size in seconds (default: 2.5)

    Returns:
        List of (raw_data_idx, start_offset_sec, end_offset_sec) tuples
    """
    # Detection window time range [start_time, end_time)
    start_time = window_idx * detection_window_sec
    end_time = start_time + detection_window_sec

    # Calculate which 2.5s windows overlap with this detection window
    # start_pos: first 2.5s window that overlaps
    # end_pos: one past the last 2.5s window that overlaps
    start_pos = int(start_time // original_window_sec)
    end_pos = int(np.ceil(end_time / original_window_sec))

    # Clamp to valid range
    start_pos = max(0, start_pos)
    end_pos = min(len(sample_indices), end_pos)

    contributing_windows = []

    for position_idx in range(start_pos, end_pos):
        raw_data_idx = sample_indices[position_idx]

        # This 2.5s window's time range
        window_start = position_idx * original_window_sec
        window_end = window_start + original_window_sec

        # Calculate overlap (guaranteed to overlap given our range calculation)
        overlap_start = max(start_time, window_start)
        overlap_end = min(end_time, window_end)

        # Convert to offsets within the 2.5s window
        offset_start = overlap_start - window_start
        offset_end = overlap_end - window_start

        contributing_windows.append((raw_data_idx, offset_start, offset_end))

    return contributing_windows


def extract_raw_data_for_window(
    contributing_windows: List[Tuple[int, float, float]],
    raw_sounds: np.ndarray,
    raw_imus: np.ndarray,
    audio_sample_rate: int = 16000,
    imu_sample_rate: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and concatenate raw sensor data for a detection window.

    Args:
        contributing_windows: [(raw_idx, start_sec, end_sec), ...]
        raw_sounds: (n_total, 1, 40000) @ 16kHz
        raw_imus: (n_total, 6, 50) @ 20Hz
        audio_sample_rate: Audio sampling rate (default: 16000)
        imu_sample_rate: IMU sampling rate (default: 20)

    Returns:
        window_audio: (1, W_audio_samples) @ 16kHz
        window_imu: (6, W_imu_samples) @ 20Hz
    """
    audio_chunks = []
    imu_chunks = []

    for raw_idx, start_sec, end_sec in contributing_windows:
        # Audio: 16kHz → 16000 samples per second
        start_sample_audio = int(start_sec * audio_sample_rate)
        end_sample_audio = int(end_sec * audio_sample_rate)
        audio_chunks.append(raw_sounds[raw_idx, :, start_sample_audio:end_sample_audio])

        # IMU: 20Hz → 20 samples per second
        start_sample_imu = int(start_sec * imu_sample_rate)
        end_sample_imu = int(end_sec * imu_sample_rate)
        imu_chunks.append(raw_imus[raw_idx, :, start_sample_imu:end_sample_imu])

    window_audio = np.concatenate(audio_chunks, axis=-1)  # (1, W*16000)
    window_imu = np.concatenate(imu_chunks, axis=-1)      # (6, W*20)

    return window_audio, window_imu


def generate_embeddings_for_detection_windows(
    sensor_data_indices: np.ndarray,
    detection_window_sec: float,
    raw_data_file: Path,
    audio_model,
    imu_model,
    fusion_model,
    device,
    original_window_sec: float = 2.5,
    batch_size: int = 2048
) -> np.ndarray:
    """
    Generate fresh embeddings for W-sized detection windows from raw sensor data.
    Uses batch processing for efficiency.

    Args:
        sensor_data_indices: (n_samples, n_original_windows) - indices to 2.5s windows
        detection_window_sec: Detection window size in seconds
        raw_data_file: Path to raw multimodal data file (lazy loaded here)
        audio_model: BEATs model
        imu_model: LIMUBert model
        fusion_model: Multimodal fusion model
        device: torch device
        original_window_sec: Original window size (default: 2.5)
        batch_size: Batch size for model inference (default: 2048)

    Returns:
        embeddings: (n_samples, n_detection_windows, 128)
    """
    n_samples, n_original_windows = sensor_data_indices.shape
    total_duration = n_original_windows * original_window_sec
    n_detection_windows = int(total_duration / detection_window_sec)

    # Load raw multimodal data (lazy loading - only when needed)
    print(f"\nLoading raw multimodal data: {raw_data_file.name}...")
    raw_multimodal_data = np.load(raw_data_file)
    raw_sounds = raw_multimodal_data['sounds']
    raw_imus = raw_multimodal_data['imus']
    print(f"  Sounds: {raw_sounds.shape}")
    print(f"  IMUs: {raw_imus.shape}")

    print(f"\nGenerating embeddings for {n_samples} samples × {n_detection_windows} windows...")
    print(f"Using inference batch size: {batch_size}")

    # Calculate sensor data dimensions
    audio_sample_size = int(detection_window_sec * 16000)
    imu_sample_size = int(detection_window_sec * 20)

    # Pre-compute extraction index matrices (same for all samples)
    print("Pre-computing detection window structure and building index matrices...")

    # Store contributing info temporarily to find max_contributing
    contributing_info = []
    for window_idx in range(n_detection_windows):
        start_time = window_idx * detection_window_sec
        end_time = start_time + detection_window_sec

        start_pos = int(start_time // original_window_sec)
        end_pos = int(np.ceil(end_time / original_window_sec))
        start_pos = max(0, start_pos)
        end_pos = min(n_original_windows, end_pos)

        chunks = []
        for position_idx in range(start_pos, end_pos):
            window_start = position_idx * original_window_sec
            window_end = window_start + original_window_sec

            overlap_start = max(start_time, window_start)
            overlap_end = min(end_time, window_end)

            offset_start = overlap_start - window_start
            offset_end = overlap_end - window_start

            chunks.append((position_idx, offset_start, offset_end))

        contributing_info.append(chunks)

    max_contributing = max(len(chunks) for chunks in contributing_info)

    print(f"  Detection windows: {n_detection_windows}")
    print(f"  Max contributing 2.5s windows per detection window: {max_contributing}")

    # Allocate and fill index matrices
    position_indices = np.full((n_detection_windows, max_contributing), -1, dtype=np.int32)
    audio_start_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
    audio_end_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
    imu_start_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
    imu_end_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
    n_contrib = np.zeros(n_detection_windows, dtype=np.int32)

    for window_idx, chunks in enumerate(contributing_info):
        n_contrib[window_idx] = len(chunks)
        for chunk_idx, (position_idx, offset_start, offset_end) in enumerate(chunks):
            position_indices[window_idx, chunk_idx] = position_idx
            audio_start_samples[window_idx, chunk_idx] = int(offset_start * 16000)
            audio_end_samples[window_idx, chunk_idx] = int(offset_end * 16000)
            imu_start_samples[window_idx, chunk_idx] = int(offset_start * 20)
            imu_end_samples[window_idx, chunk_idx] = int(offset_end * 20)

    # Process in sample batches to avoid memory overflow
    # For 10k samples: 20GB+ if done all at once, <2GB if done in batches of 500
    sample_batch_size = 500
    num_sample_batches = (n_samples + sample_batch_size - 1) // sample_batch_size

    print(f"Processing {n_samples} samples in {num_sample_batches} batches of up to {sample_batch_size} samples each...")
    print(f"This avoids loading all {n_samples}×{n_detection_windows} windows into memory at once.\n")

    # Allocate final output array
    embeddings = np.zeros((n_samples, n_detection_windows, 128), dtype=np.float32)

    import time
    start_time = time.time()

    for sb_idx in range(num_sample_batches):
        sb_start = sb_idx * sample_batch_size
        sb_end = min(sb_start + sample_batch_size, n_samples)
        sb_size = sb_end - sb_start

        print(f"{'='*80}")
        print(f"SAMPLE BATCH {sb_idx + 1}/{num_sample_batches}: Samples {sb_start}-{sb_end-1} ({sb_size} samples)")
        print(f"{'='*80}")

        # Allocate memory for current sample batch only
        batch_audio_windows = np.zeros((sb_size, n_detection_windows, 1, audio_sample_size), dtype=np.float32)
        batch_imu_windows = np.zeros((sb_size, n_detection_windows, 6, imu_sample_size), dtype=np.float32)

        # Extract raw sensor data for current sample batch
        print(f"[1/2] Extracting raw sensor data for samples {sb_start}-{sb_end-1}...")
        for local_idx in range(sb_size):
            sample_idx = sb_start + local_idx

            if (local_idx + 1) % 100 == 0 or local_idx == 0:
                print(f"  Sample {sample_idx + 1}/{n_samples}...")

            for window_idx in range(n_detection_windows):
                audio_chunks = []
                imu_chunks = []

                # Use pre-computed indices
                for chunk_idx in range(n_contrib[window_idx]):
                    pos_idx = position_indices[window_idx, chunk_idx]
                    raw_data_idx = sensor_data_indices[sample_idx, pos_idx]

                    # Audio extraction
                    audio_start = audio_start_samples[window_idx, chunk_idx]
                    audio_end = audio_end_samples[window_idx, chunk_idx]
                    audio_chunks.append(raw_sounds[raw_data_idx, :, audio_start:audio_end])

                    # IMU extraction
                    imu_start = imu_start_samples[window_idx, chunk_idx]
                    imu_end = imu_end_samples[window_idx, chunk_idx]
                    imu_chunks.append(raw_imus[raw_data_idx, :, imu_start:imu_end])

                concatenated_audio = np.concatenate(audio_chunks, axis=-1)
                concatenated_imu = np.concatenate(imu_chunks, axis=-1)

                # Pad or trim to match expected size
                if concatenated_audio.shape[-1] != audio_sample_size:
                    if concatenated_audio.shape[-1] < audio_sample_size:
                        pad_size = audio_sample_size - concatenated_audio.shape[-1]
                        concatenated_audio = np.pad(concatenated_audio, ((0, 0), (0, pad_size)), mode='constant')
                    else:
                        concatenated_audio = concatenated_audio[:, :audio_sample_size]

                if concatenated_imu.shape[-1] != imu_sample_size:
                    if concatenated_imu.shape[-1] < imu_sample_size:
                        pad_size = imu_sample_size - concatenated_imu.shape[-1]
                        concatenated_imu = np.pad(concatenated_imu, ((0, 0), (0, pad_size)), mode='constant')
                    else:
                        concatenated_imu = concatenated_imu[:, :imu_sample_size]

                batch_audio_windows[local_idx, window_idx] = concatenated_audio
                batch_imu_windows[local_idx, window_idx] = concatenated_imu

        # Reshape to (sb_size * n_detection_windows, ...)
        batch_audio_flat = batch_audio_windows.reshape(-1, audio_sample_size)
        batch_imu_flat = batch_imu_windows.reshape(-1, 6, imu_sample_size)

        # Free the large arrays immediately
        del batch_audio_windows, batch_imu_windows

        total_windows_batch = sb_size * n_detection_windows
        batch_embeddings_flat = np.zeros((total_windows_batch, 128), dtype=np.float32)

        # Generate embeddings for current sample batch
        print(f"[2/2] Generating embeddings for {total_windows_batch} windows in batches of {batch_size}...")

        with torch.no_grad():
            for emb_start in range(0, total_windows_batch, batch_size):
                emb_end = min(emb_start + batch_size, total_windows_batch)

                if (emb_start // batch_size + 1) % 10 == 0 or emb_start == 0:
                    elapsed = time.time() - start_time
                    total_windows_done = sb_start * n_detection_windows + emb_start
                    total_all_windows = n_samples * n_detection_windows
                    if total_windows_done > 0:
                        rate = total_windows_done / elapsed
                        remaining = (total_all_windows - total_windows_done) / rate
                        print(f"  Embedding batch {emb_start // batch_size + 1}/{(total_windows_batch + batch_size - 1) // batch_size} - Overall: {total_windows_done}/{total_all_windows} ({rate:.0f} windows/s, ETA: {remaining/60:.1f} min)")
                    else:
                        print(f"  Embedding batch {emb_start // batch_size + 1}/{(total_windows_batch + batch_size - 1) // batch_size}")

                # Convert batch to tensors and move to device
                audio_batch = torch.from_numpy(batch_audio_flat[emb_start:emb_end]).float().to(device)
                imu_batch = torch.from_numpy(batch_imu_flat[emb_start:emb_end]).float().to(device)

                # Create padding mask for BEATs
                padding_mask = torch.zeros_like(audio_batch, dtype=torch.bool)

                # LIMUBert expects (batch, seq_len, channels), permute IMU data
                imu_batch = imu_batch.permute(0, 2, 1)

                # BEATs audio embedding
                audio_emb = audio_model.extract_features(audio_batch, padding_mask=padding_mask)[0]

                # LIMUBert IMU embedding
                imu_emb = imu_model(imu_batch)

                # Fusion
                fused_emb = fusion_model.extract_features(audio_emb, imu_emb)

                batch_embeddings_flat[emb_start:emb_end] = fused_emb.cpu().numpy()

        # Store embeddings for current sample batch
        embeddings[sb_start:sb_end] = batch_embeddings_flat.reshape(sb_size, n_detection_windows, 128)

        # Free memory
        del batch_audio_flat, batch_imu_flat, batch_embeddings_flat

        print(f"Completed sample batch {sb_idx + 1}/{num_sample_batches}\n")

    print(f"Embedding generation complete: {embeddings.shape}")
    return embeddings


def recompute_ce_labels(
    aggregated_ae_labels: np.ndarray,
    detection_window_sec: float
) -> np.ndarray:
    """
    Re-compute CE labels by feeding noisy AE sequences through FSMs.

    Args:
        aggregated_ae_labels: Aggregated AE labels after majority voting, shape (n_samples, n_windows)
        detection_window_sec: Duration of each detection window in seconds

    Returns:
        ce_labels: Multi-label CE labels, shape (n_samples, n_windows, 10)
    """
    n_samples, n_windows = aggregated_ae_labels.shape
    ce_labels = np.zeros((n_samples, n_windows, 10), dtype=np.int8)

    # AE label to action name mapping
    label_to_action = {
        0: 'brush_teeth',
        1: 'click_mouse',
        2: 'drink',
        3: 'eat',
        4: 'flush_toilet',
        5: 'sit',
        6: 'type',
        7: 'walk',
        8: 'wash'
    }

    print(f"\nRe-computing CE labels for {n_samples} samples with {n_windows} windows each...")

    for sample_idx in range(n_samples):
        # Initialize FSMs for this sample with detection window size
        fsm_list = [
            Event1FSM(window_sec=detection_window_sec),
            Event2FSM(window_sec=detection_window_sec),
            Event3FSM(window_sec=detection_window_sec),
            Event4FSM(window_sec=detection_window_sec),
            Event5FSM(window_sec=detection_window_sec),
            Event6FSM(window_sec=detection_window_sec),
            Event7FSM(window_sec=detection_window_sec),
            Event8FSM(window_sec=detection_window_sec),
            Event9FSM(window_sec=detection_window_sec),
            Event10FSM(window_sec=detection_window_sec),
        ]

        # Feed AE sequence through FSMs
        for window_idx in range(n_windows):
            ae_label = aggregated_ae_labels[sample_idx, window_idx]
            action = label_to_action[ae_label]

            # Update all FSMs and collect CE labels
            ce_label_vector = np.zeros(10, dtype=np.int8)
            for fsm in fsm_list:
                event_label = fsm.update_state(input=action)
                if event_label > 0:
                    ce_label_vector[event_label - 1] = 1  # Convert 1-10 to 0-9

            ce_labels[sample_idx, window_idx, :] = ce_label_vector

        if (sample_idx + 1) % 100 == 0:
            print(f"  Processed {sample_idx + 1}/{n_samples} samples")

    print("CE label re-computation complete")

    return ce_labels


def process_nar_dataset(
    input_file: Path,
    detection_window_sec: float,
    original_window_sec: float = 2.5,
    delta_sec: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process NAR dataset with realistic windowing (no embeddings).

    Args:
        input_file: Path to input NAR data file (AE labels)
        detection_window_sec: Target detection window size in seconds
        original_window_sec: Original window size (default: 2.5)
        delta_sec: Delta granularity (default: 0.5)

    Returns:
        aggregated_ae: AE labels after majority voting, shape (n_samples, n_new_windows)
        ce_labels: Re-computed CE labels, shape (n_samples, n_new_windows, 10)
    """
    print("\n" + "="*80)
    print("REALISTIC DETECTION WINDOW PROCESSING (NAR)")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Detection window size: {detection_window_sec}s")
    print(f"Delta granularity: {delta_sec}s")
    print("="*80 + "\n")

    # Load NAR data (AE labels only)
    print("[1/4] Loading NAR data...")
    ae_labels = np.load(input_file)
    n_samples, n_windows_baseline = ae_labels.shape
    print(f"Loaded {n_samples} samples with {n_windows_baseline} windows each")

    # Step 2: Subdivide AE labels to deltas
    print("\n[2/4] Subdividing AE labels to fine-grained deltas...")
    ae_deltas = subdivide_to_deltas(ae_labels, original_window_sec, delta_sec)

    # Step 3: Aggregate AE with majority voting
    print("\n[3/4] Aggregating AE labels with majority voting...")
    aggregated_ae = aggregate_with_majority_vote(ae_deltas, detection_window_sec, delta_sec)

    # Step 4: Re-compute CE labels
    print("\n[4/4] Re-computing CE labels with FSMs...")
    ce_labels = recompute_ce_labels(aggregated_ae, detection_window_sec)

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output AE labels shape: {aggregated_ae.shape}")
    print(f"Output CE labels shape: {ce_labels.shape}")
    print("="*80 + "\n")

    return aggregated_ae, ce_labels


def process_standard_dataset(
    index_file_path: Path,
    detection_window_sec: float,
    raw_data_file: Path,
    audio_model,
    imu_model,
    fusion_model,
    device,
    original_window_sec: float = 2.5,
    delta_sec: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process standard dataset with fresh embedding generation.

    Args:
        index_file_path: Path to index .npz file
        detection_window_sec: Target detection window size in seconds
        raw_data_file: Path to raw multimodal data file (lazy loaded)
        audio_model: BEATs model
        imu_model: LIMUBert model
        fusion_model: Fusion model
        device: torch device
        original_window_sec: Original window size (default: 2.5)
        delta_sec: Delta granularity (default: 0.5)

    Returns:
        embeddings: Fresh embeddings for detection windows, shape (n_samples, n_new_windows, 128)
        noisy_ae_labels: AE labels after majority voting, shape (n_samples, n_new_windows)
        ce_labels: Re-computed CE labels, shape (n_samples, n_new_windows, 10)
    """
    print("\n" + "="*80)
    print("REALISTIC DETECTION WINDOW PROCESSING")
    print("="*80)
    print(f"Index file: {index_file_path}")
    print(f"Detection window size: {detection_window_sec}s")
    print(f"Delta granularity: {delta_sec}s")
    print("="*80 + "\n")

    # Load index file
    print("[1/5] Loading index file...")
    index_data = np.load(index_file_path, allow_pickle=True)
    sensor_data_indices = index_data['sensor_data_indices']
    action_labels = index_data['action_labels']

    print(f"Loaded {sensor_data_indices.shape[0]} samples with {sensor_data_indices.shape[1]} windows each")

    # Step 2: Subdivide AE labels to deltas
    print("\n[2/5] Subdividing AE labels to fine-grained deltas...")
    ae_deltas = subdivide_to_deltas(action_labels, original_window_sec, delta_sec)

    # Step 3: Aggregate AE with majority voting
    print("\n[3/5] Aggregating AE labels with majority voting...")
    noisy_ae_labels = aggregate_with_majority_vote(ae_deltas, detection_window_sec, delta_sec)

    # Step 4: Generate fresh embeddings for detection windows
    print("\n[4/5] Generating fresh embeddings for detection windows...")
    embeddings = generate_embeddings_for_detection_windows(
        sensor_data_indices,
        detection_window_sec,
        raw_data_file,  # Pass Path for lazy loading
        audio_model,
        imu_model,
        fusion_model,
        device,
        original_window_sec
    )

    # Step 5: Re-compute CE labels
    print("\n[5/5] Re-computing CE labels with FSMs...")
    ce_labels = recompute_ce_labels(noisy_ae_labels, detection_window_sec)

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output AE labels shape: {noisy_ae_labels.shape}")
    print(f"Output CE labels shape: {ce_labels.shape}")
    print("="*80 + "\n")

    return embeddings, noisy_ae_labels, ce_labels


def process_ae2ce_dataset(
    input_embed_file: Path,
    input_labels_file: Path,
    ae_classifier,
    device,
    soft: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process existing realistic windowing embeddings to generate AE2CE datasets.

    Args:
        input_embed_file: Path to existing embeddings file (128-dim)
        input_labels_file: Path to existing CE labels file
        ae_classifier: Loaded AE classifier model
        device: torch device
        soft: If True, generate soft probabilities; if False, generate one-hot

    Returns:
        ae2ce_embeddings: AE2CE embeddings, shape (n_samples, n_windows, 9)
        ce_labels: CE labels (unchanged), shape (n_samples, n_windows, 10)
    """
    print("\n" + "="*80)
    print("AE2CE DATASET GENERATION FROM REALISTIC WINDOWING")
    print("="*80)
    print(f"Input embeddings: {input_embed_file}")
    print(f"Input labels: {input_labels_file}")
    print(f"Mode: {'Soft probabilities' if soft else 'One-hot predictions'}")
    print("="*80 + "\n")

    # Load existing embeddings
    print("[1/2] Loading existing realistic windowing embeddings...")
    embeddings = np.load(input_embed_file)
    ce_labels = np.load(input_labels_file)

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {ce_labels.shape}")

    # Validate shape
    if embeddings.shape[2] != 128:
        print(f"\nERROR: Expected 128-dim embeddings, got {embeddings.shape[2]}-dim")
        print(f"Make sure the input file contains fusion embeddings, not AE2CE data")
        sys.exit(1)

    # Generate AE2CE
    print("\n[2/2] Generating AE2CE embeddings...")

    # Reshape embeddings from (n_samples, n_windows, 128) to (n_samples * n_windows, 128)
    n_samples, n_windows, embed_dim = embeddings.shape
    embeddings_flat = embeddings.reshape(-1, embed_dim)  # (n_samples * n_windows, 128)

    ae2ce_embeddings_flat = generate_ae2ce_from_fusion(
        embeddings_flat,
        ae_classifier,
        device=device,
        soft=soft
    )

    # Reshape back to (n_samples, n_windows, 9)
    ae2ce_embeddings = ae2ce_embeddings_flat.reshape(n_samples, n_windows, 9)

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output AE2CE shape: {ae2ce_embeddings.shape}")
    print(f"Output CE labels shape: {ce_labels.shape}")
    print("="*80 + "\n")

    return ae2ce_embeddings, ce_labels


def main():
    parser = argparse.ArgumentParser(
        description='Convert fixed-window (2.5s) CE datasets to realistic variable-window datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard dataset - Train split
  python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split train --n-samples 2000 --gpu-id 0

  # Standard dataset - Val split
  python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split val --gpu-id 0

  # NAR dataset - Train split
  python realistic_windowing.py --dataset-type nar --input-dir CE_dataset/nar --detection-window 2.0 --output-dir CE_dataset/nar_w2.0s --split train --n-samples 40000

  # NAR dataset - Val split
  python realistic_windowing.py --dataset-type nar --input-dir CE_dataset/nar --detection-window 2.0 --output-dir CE_dataset/nar_w2.0s --split val

  # Generate AE2CE from existing realistic windowing embeddings - Train split
  python realistic_windowing.py --dataset-type standard --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split train --n-samples 2000 --ae2ce

  # Generate soft AE2CE from existing realistic windowing embeddings - Val split
  python realistic_windowing.py --dataset-type standard --detection-window 2.0 --output-dir CE_dataset/realistic_w2.0s --split val --soft-ae2ce
        """
    )

    parser.add_argument('--dataset-type', type=str, required=True, choices=['standard', 'nar'],
                        help='Dataset type: standard (with embeddings) or nar (symbolic labels only)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing index files (standard) or NAR data files (nar)')
    parser.add_argument('--detection-window', type=float, required=True,
                        help='Detection window size in seconds (must be multiple of 0.5)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for realistic datasets')
    parser.add_argument('--raw-data-dir', type=str,
                        help='Directory containing raw multimodal data (required for standard datasets)')
    parser.add_argument('--original-window', type=float, default=2.5,
                        help='Original window size in seconds (default: 2.5)')
    parser.add_argument('--delta', type=float, default=0.5,
                        help='Delta granularity in seconds (default: 0.5)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID (default: 0, only used for standard datasets)')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split to process (train, val, or test)')
    parser.add_argument('--duration', type=int, default=5, choices=[3, 5, 15, 30],
                        help='CE duration in minutes (default: 5)')
    parser.add_argument('--n-samples', type=int,
                        help='Number of samples (required for train split)')
    parser.add_argument('--rule-only', action='store_true',
                        help='Use rule-only test set (only valid for test split with standard datasets)')
    parser.add_argument('--ae2ce', action='store_true',
                        help='Generate ae2ce dataset (one-hot AE predictions) from existing realistic embeddings')
    parser.add_argument('--soft-ae2ce', action='store_true',
                        help='Generate soft-ae2ce dataset (soft probability AE predictions) from existing realistic embeddings')

    args = parser.parse_args()

    # Validate arguments
    if args.detection_window % args.delta != 0:
        print(f"ERROR: Detection window ({args.detection_window}s) must be a multiple of delta ({args.delta}s)")
        sys.exit(1)

    if args.duration != 5 and args.split != 'test':
        print(f"ERROR: --duration other than 5 can only be used with --split test (got: --duration {args.duration} with --split {args.split})")
        sys.exit(1)

    if args.rule_only and args.split != 'test':
        print("ERROR: --rule-only can only be used with --split test")
        sys.exit(1)

    if args.rule_only and args.dataset_type == 'nar':
        print("ERROR: --rule-only is only valid for standard datasets")
        sys.exit(1)

    if args.split == 'train' and args.n_samples is None:
        print("ERROR: --n-samples is required for train split")
        sys.exit(1)

    if args.dataset_type == 'standard' and args.raw_data_dir is None and not (args.ae2ce or args.soft_ae2ce):
        print("ERROR: --raw-data-dir is required for standard datasets (not needed for ae2ce generation)")
        sys.exit(1)

    if args.ae2ce and args.soft_ae2ce:
        print("ERROR: --ae2ce and --soft-ae2ce cannot be used together")
        sys.exit(1)

    if (args.ae2ce or args.soft_ae2ce) and args.dataset_type != 'standard':
        print("ERROR: --ae2ce and --soft-ae2ce can only be used with --dataset-type standard")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    # Route to appropriate processing function
    split = args.split

    if args.dataset_type == 'nar':
        # ========== NAR DATASET PROCESSING ==========
        print(f"\n{'='*80}")
        print(f"REALISTIC WINDOWING - NAR DATASET")
        print(f"{'='*80}")
        print(f"Dataset type:     NAR (symbolic labels only)")
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Detection window: {args.detection_window}s")
        print(f"Split:            {split}")
        if split == 'train':
            print(f"N samples:        {args.n_samples}")
        print(f"{'='*80}\n")

        # Determine input file based on split and duration
        if split == 'train':
            input_file = input_dir / f"ct_ce{args.duration}min_train_data_{args.n_samples}.npy"
        elif split == 'val':
            input_file = input_dir / f"ct_ce{args.duration}min_val_data.npy"
        else:  # test
            input_file = input_dir / f"ct_ce{args.duration}min_test_data.npy"

        if not input_file.exists():
            print(f"ERROR: Input file not found: {input_file}")
            sys.exit(1)

        # Process NAR dataset
        aggregated_ae, ce_labels = process_nar_dataset(
            input_file,
            args.detection_window,
            args.original_window,
            args.delta
        )

        # Save outputs
        if split == 'train':
            output_data_file = output_dir / f"ct_ce{args.duration}min_train_data_{args.n_samples}.npy"
            output_labels_file = output_dir / f"ct_ce{args.duration}min_train_labels_{args.n_samples}.npy"
        elif split == 'val':
            output_data_file = output_dir / f"ct_ce{args.duration}min_val_data.npy"
            output_labels_file = output_dir / f"ct_ce{args.duration}min_val_labels.npy"
        else:  # test
            output_data_file = output_dir / f"ct_ce{args.duration}min_test_data.npy"
            output_labels_file = output_dir / f"ct_ce{args.duration}min_test_labels.npy"

        np.save(output_data_file, aggregated_ae)
        np.save(output_labels_file, ce_labels)

        print(f"\n{'='*80}")
        print("ALL PROCESSING COMPLETE")
        print(f"{'='*80}")
        print("Saved to:")
        print(f"  {output_data_file}")
        print(f"  {output_labels_file}")
        print(f"{'='*80}\n")

    else:
        # ========== STANDARD DATASET PROCESSING ==========
        # Load models
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        # Check if we're doing AE2CE generation
        if args.ae2ce or args.soft_ae2ce:
            # ========== AE2CE GENERATION FROM EXISTING REALISTIC EMBEDDINGS ==========
            print(f"\n{'='*80}")
            print(f"REALISTIC WINDOWING - AE2CE GENERATION")
            print(f"{'='*80}")
            print(f"Mode:             {'Soft AE2CE' if args.soft_ae2ce else 'One-hot AE2CE'}")
            print(f"Device:           {device}")
            print(f"Detection window: {args.detection_window}s")
            print(f"Split:            {split}")
            if split == 'train':
                print(f"N samples:        {args.n_samples}")
            print(f"{'='*80}\n")

            # Determine input filenames (existing realistic embeddings)
            if split == 'train':
                input_embed_file = output_dir / f"ce{args.duration}min_train_data_{args.n_samples}.npy"
                input_labels_file = output_dir / f"ce{args.duration}min_train_labels_{args.n_samples}.npy"
            elif args.rule_only:
                input_embed_file = output_dir / f"ce{args.duration}min_test_rule_data.npy"
                input_labels_file = output_dir / f"ce{args.duration}min_test_rule_labels.npy"
            else:
                input_embed_file = output_dir / f"ce{args.duration}min_{split}_data.npy"
                input_labels_file = output_dir / f"ce{args.duration}min_{split}_labels.npy"

            # Check if input files exist
            if not input_embed_file.exists():
                print(f"ERROR: Input embeddings file not found: {input_embed_file}")
                print(f"Please generate realistic windowing embeddings first using:")
                print(f"  python realistic_windowing.py --dataset-type standard --input-dir CE_dataset --raw-data-dir Multimodal --detection-window {args.detection_window} --output-dir {args.output_dir} --split {split}" + (f" --n-samples {args.n_samples}" if split == 'train' else ""))
                sys.exit(1)

            if not input_labels_file.exists():
                print(f"ERROR: Input labels file not found: {input_labels_file}")
                sys.exit(1)

            # Load AE classifier
            print("Loading AE classifier...")
            ae_classifier = load_ae_classifier(device=device)

            # Process AE2CE
            ae2ce_embeddings, ce_labels = process_ae2ce_dataset(
                input_embed_file,
                input_labels_file,
                ae_classifier,
                device,
                soft=args.soft_ae2ce
            )

            # Save outputs - use same format as dataset_loader.py
            if args.soft_ae2ce:
                dataset_prefix = f"softae2ce{args.duration}min"
            else:
                dataset_prefix = f"ae2ce{args.duration}min"

            if split == 'train':
                output_data_file = output_dir / f"{dataset_prefix}_train_data_{args.n_samples}.npy"
                output_labels_file = output_dir / f"{dataset_prefix}_train_labels_{args.n_samples}.npy"
            elif args.rule_only:
                output_data_file = output_dir / f"{dataset_prefix}_test_rule_data.npy"
                output_labels_file = output_dir / f"{dataset_prefix}_test_rule_labels.npy"
            else:
                output_data_file = output_dir / f"{dataset_prefix}_{split}_data.npy"
                output_labels_file = output_dir / f"{dataset_prefix}_{split}_labels.npy"

            np.save(output_data_file, ae2ce_embeddings)
            np.save(output_labels_file, ce_labels)

            print(f"\n{'='*80}")
            print("ALL PROCESSING COMPLETE")
            print(f"{'='*80}")
            print("Saved to:")
            print(f"  {output_data_file}")
            print(f"  {output_labels_file}")
            print(f"{'='*80}\n")

        else:
            # ========== STANDARD EMBEDDING GENERATION ==========
            print(f"\n{'='*80}")
            print(f"REALISTIC WINDOWING - STANDARD DATASET")
            print(f"{'='*80}")
            print(f"Dataset type:     Standard (with embeddings)")
            print(f"Device:           {device}")
            print(f"Detection window: {args.detection_window}s")
            print(f"{'='*80}\n")

            # Load audio and IMU models (window-agnostic)
            print("Loading models...")
            audio_model = load_beats_model(device=device)
            imu_model = load_limubert_model(device=device)

            # Load window-specific fusion model
            fusion_model_path = f'./saved_models/multimodal_embed_model_w{args.detection_window}.pt'
            fusion_model = load_multimodal_fusion_model(device=device, model_path=fusion_model_path)

            # Determine index filename based on split and duration
            if split == 'train':
                index_file = input_dir / f"ce{args.duration}min_train_indices_{args.n_samples}.npz"
            elif split == 'test' and args.rule_only:
                index_file = input_dir / f"ce{args.duration}min_test_rule_indices.npz"
            else:
                index_file = input_dir / f"ce{args.duration}min_{split}_indices.npz"
    
            if not index_file.exists():
                print(f"ERROR: Index file not found: {index_file}")
                sys.exit(1)
    
            print(f"\n{'#'*80}")
            print(f"# Processing {split.upper()} split{' (rule-only)' if args.rule_only else ''}")
            print(f"{'#'*80}")
            print(f"Index file: {index_file}")
    
            # Load index config to determine which raw data file to use
            print("\n[0/5] Loading index config...")
            index_data = np.load(index_file, allow_pickle=True)
            config = index_data['config'].item()
            # Parse JSON string if needed
            if isinstance(config, str):
                config = json.loads(config)
            data_cat = config['data_cat']  # 'train' or 'test'
            print(f"  Data category: {data_cat} ({'subjects 1-4' if data_cat == 'train' else 'subject 5'})")
    
            # Determine raw data file path (don't load yet - will load lazily during embedding generation)
            raw_data_dir = Path(args.raw_data_dir)
            if data_cat == 'train':
                raw_data_file = raw_data_dir / 'MultimodalDataset_audio-40000_1234_imu-50_1234.npz'
            else:  # data_cat == 'test'
                raw_data_file = raw_data_dir / 'MultimodalDataset_audio-40000_5_imu-50_5.npz'
    
            if not raw_data_file.exists():
                print(f"ERROR: Raw data file not found: {raw_data_file}")
                sys.exit(1)
    
            print(f"  Raw data file: {raw_data_file.name}")
            print(f"  (Will load during embedding generation...)")
    
            # Process dataset - pass raw_data_file path for lazy loading
            embeddings, noisy_ae_labels, ce_labels = process_standard_dataset(
                index_file,
                args.detection_window,
                raw_data_file,  # Pass Path, not loaded data
                audio_model,
                imu_model,
                fusion_model,
                device,
                args.original_window,
                args.delta
            )
    
            # Save outputs
            if split == 'train':
                output_embed_file = output_dir / f"ce{args.duration}min_train_data_{args.n_samples}.npy"
                output_ae_file = output_dir / f"ce{args.duration}min_train_ae_{args.n_samples}.npy"
                output_ce_file = output_dir / f"ce{args.duration}min_train_labels_{args.n_samples}.npy"
            elif args.rule_only:
                output_embed_file = output_dir / f"ce{args.duration}min_test_rule_data.npy"
                output_ae_file = output_dir / f"ce{args.duration}min_test_rule_ae.npy"
                output_ce_file = output_dir / f"ce{args.duration}min_test_rule_labels.npy"
            else:
                output_embed_file = output_dir / f"ce{args.duration}min_{split}_data.npy"
                output_ae_file = output_dir / f"ce{args.duration}min_{split}_ae.npy"
                output_ce_file = output_dir / f"ce{args.duration}min_{split}_labels.npy"
    
            np.save(output_embed_file, embeddings)
            np.save(output_ae_file, noisy_ae_labels)
            np.save(output_ce_file, ce_labels)
    
            print(f"\n{'='*80}")
            print("ALL PROCESSING COMPLETE")
            print(f"{'='*80}")
            print("Saved to:")
            print(f"  {output_embed_file}")
            print(f"  {output_ae_file}")
            print(f"  {output_ce_file}")
            print(f"{'='*80}\n")


if __name__ == '__main__':
    # Quick test for detection window extraction logic
    if len(sys.argv) > 1 and sys.argv[1] == '--test-mapping':
        print("="*80)
        print("Testing Detection Window Extraction (Index Matrix Method)")
        print("="*80)

        # Test with different window sizes
        test_configs = [
            (1.0, "W < 2.5s (smaller than original)"),
            (2.0, "W < 2.5s (smaller than original)"),
            (2.5, "W = 2.5s (exact match)"),
            (3.0, "W > 2.5s (larger than original)"),
            (5.0, "W > 2.5s (much larger)"),
        ]

        original_window_sec = 2.5
        n_original_windows = 20  # Simulate 50s of data (20 × 2.5s)

        for detection_window_sec, description in test_configs:
            print(f"\n{'='*80}")
            print(f"TEST: Detection window = {detection_window_sec}s ({description})")
            print(f"{'='*80}")

            total_duration = n_original_windows * original_window_sec
            n_detection_windows = int(total_duration / detection_window_sec)

            print(f"Original windows: {n_original_windows} × {original_window_sec}s = {total_duration}s")
            print(f"Detection windows: {n_detection_windows} × {detection_window_sec}s")

            # Build extraction structure (same as in the main code)
            max_contributing = 0
            contributing_list = []

            for window_idx in range(n_detection_windows):
                start_time = window_idx * detection_window_sec
                end_time = start_time + detection_window_sec

                start_pos = int(start_time // original_window_sec)
                end_pos = int(np.ceil(end_time / original_window_sec))
                start_pos = max(0, start_pos)
                end_pos = min(n_original_windows, end_pos)

                contributing = []
                for position_idx in range(start_pos, end_pos):
                    window_start = position_idx * original_window_sec
                    window_end = window_start + original_window_sec

                    overlap_start = max(start_time, window_start)
                    overlap_end = min(end_time, window_end)

                    offset_start = overlap_start - window_start
                    offset_end = overlap_end - window_start

                    contributing.append((position_idx, offset_start, offset_end))

                contributing_list.append(contributing)
                max_contributing = max(max_contributing, len(contributing))

            print(f"Max contributing 2.5s windows per detection window: {max_contributing}")

            # Build index matrices (same as main code)
            position_indices = np.full((n_detection_windows, max_contributing), -1, dtype=np.int32)
            audio_start_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
            audio_end_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
            imu_start_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
            imu_end_samples = np.zeros((n_detection_windows, max_contributing), dtype=np.int32)
            n_contrib = np.zeros(n_detection_windows, dtype=np.int32)

            for window_idx, contributing in enumerate(contributing_list):
                n_contrib[window_idx] = len(contributing)
                for chunk_idx, (position_idx, offset_start, offset_end) in enumerate(contributing):
                    position_indices[window_idx, chunk_idx] = position_idx
                    audio_start_samples[window_idx, chunk_idx] = int(offset_start * 16000)
                    audio_end_samples[window_idx, chunk_idx] = int(offset_end * 16000)
                    imu_start_samples[window_idx, chunk_idx] = int(offset_start * 20)
                    imu_end_samples[window_idx, chunk_idx] = int(offset_end * 20)

            print("\nIndex Matrices:")
            print(f"  position_indices shape: {position_indices.shape}")
            print(f"  audio_start_samples shape: {audio_start_samples.shape}")
            print(f"  n_contrib shape: {n_contrib.shape}")

            # Show first few detection windows with index matrices
            print("\nFirst 3 detection windows (with index matrices):")
            for window_idx in range(min(3, n_detection_windows)):
                start_time = window_idx * detection_window_sec
                end_time = start_time + detection_window_sec

                print(f"\n  Detection Window {window_idx}: [{start_time:.1f}s - {end_time:.1f}s)")
                print(f"  n_contrib[{window_idx}] = {n_contrib[window_idx]}")

                total_duration = 0
                for chunk_idx in range(n_contrib[window_idx]):
                    pos_idx = position_indices[window_idx, chunk_idx]
                    audio_start = audio_start_samples[window_idx, chunk_idx]
                    audio_end = audio_end_samples[window_idx, chunk_idx]
                    imu_start = imu_start_samples[window_idx, chunk_idx]
                    imu_end = imu_end_samples[window_idx, chunk_idx]

                    duration_sec = (audio_end - audio_start) / 16000
                    total_duration += duration_sec

                    print(f"    Chunk {chunk_idx}:")
                    print(f"      position_indices[{window_idx},{chunk_idx}] = {pos_idx}")
                    print(f"      audio: samples [{audio_start}:{audio_end}] = {audio_end - audio_start} samples = {duration_sec:.2f}s")
                    print(f"      imu:   samples [{imu_start}:{imu_end}] = {imu_end - imu_start} samples")

                print(f"  Total duration: {total_duration:.2f}s (expected: {detection_window_sec}s) {'✓' if abs(total_duration - detection_window_sec) < 0.01 else '✗'}")

            # Show how it would work with actual sample_indices
            print(f"\n  Example with sample_indices = [145, 289, 301, 450, ...]:")
            sample_indices_example = np.array([145, 289, 301, 450, 502, 600, 701, 802, 903, 1004,
                                               1105, 1206, 1307, 1408, 1509, 1610, 1711, 1812, 1913, 2014])

            for window_idx in range(min(2, n_detection_windows)):
                start_time = window_idx * detection_window_sec
                end_time = start_time + detection_window_sec
                print(f"\n    Detection Window {window_idx} [{start_time:.1f}s - {end_time:.1f}s):")

                for chunk_idx in range(n_contrib[window_idx]):
                    pos_idx = position_indices[window_idx, chunk_idx]
                    raw_data_idx = sample_indices_example[pos_idx]
                    audio_start = audio_start_samples[window_idx, chunk_idx]
                    audio_end = audio_end_samples[window_idx, chunk_idx]

                    print(f"      Chunk {chunk_idx}: raw_sounds[{raw_data_idx}, :, {audio_start}:{audio_end}]")

        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80 + "\n")
        sys.exit(0)

    main()
