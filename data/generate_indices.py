#!/usr/bin/env python3
"""
CE Index File Generator - CLI Tool

Generates lightweight index files for Complex Event datasets.
Index files specify which raw sensor samples to use for each CE sample.

Usage:
    python generate_indices.py --duration 5 --split train --n_samples 2000
    python generate_indices.py --duration 5 --split val --n_samples 500
    python generate_indices.py --duration 15 --split test --n_samples 200
"""

import argparse
import os
import sys
from pathlib import Path
from ce_generator import CE3min, CE5min, CE15min, CE30min


# Hyperparameters
WINDOW_SIZE = 2.5  # Window size in seconds (must match raw multimodal data)

# Map duration to generator class
DURATION_TO_CLASS = {
    3: CE3min,
    5: CE5min,
    15: CE15min,
    30: CE30min
}

# Output directory for index files
OUTPUT_DIR = './CE_dataset/'


def main():
    parser = argparse.ArgumentParser(
        description='Generate CE index files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training index file with 2000 samples (5min duration)
  python generate_indices.py --duration 5 --split train --n_samples 2000
  Output: CE_dataset/ce5min_train_indices.npz (~0.07 MB)

  # Generate validation index file with 500 samples (always 5min)
  python generate_indices.py --duration 5 --split val --n_samples 500
  Output: CE_dataset/ce5min_val_indices.npz

  # Test type 1: Subject + temporal generalization (subject 5, test split)
  python generate_indices.py --duration 15 --split test --n_samples 200
  Output: CE_dataset/ce15min_test_indices.npz (subject 5, new temporal patterns)

  # Test type 2: Rule/pattern generalization only (subjects 1-4, train split)
  python generate_indices.py --duration 15 --split test --n_samples 200 --rule-only
  Output: CE_dataset/ce15min_test_rule_indices.npz (same subjects, new CE combinations)

Workflow:
  Step 1: Generate index files (this script)
          - Lightweight (~0.07 MB per 100 samples)
          - Fast generation (no GPU needed)
          - Defines CE sample structure

  Step 2: Generate embeddings (dataset_loader.py)
          - Reads index files
          - Generates embeddings using BEATs + LIMUBert + Fusion
          - Outputs data + labels (.npy files)

Index file contains:
  - sensor_data_indices: Which raw samples to use
  - ce_labels: Complex event labels (0-10)
  - actions: Action names per window
  - action_labels: Action label IDs (0-8)
  - data_sources: 'train' or 'test' split
  - config: Generation metadata
        """
    )

    parser.add_argument(
        '--duration',
        type=int,
        required=True,
        choices=[3, 5, 15, 30],
        help='CE duration in minutes (3, 5, 15, or 30)'
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val', 'test'],
        help='Dataset split (train, val, or test)'
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        required=True,
        help='Number of CE samples to generate (e.g., 2000 for train, 500 for val)'
    )

    parser.add_argument(
        '--positive_only',
        action='store_true',
        help='Generate only samples with CE violations (for testing)'
    )

    parser.add_argument(
        '--rule-only',
        action='store_true',
        help='Test rule/pattern generalization only (uses train split, adds _rule suffix)'
    )

    args = parser.parse_args()

    # Validate flag usage
    if args.rule_only and args.split != 'test':
        parser.error("--rule-only can only be used with --split test")

    # Validate split conventions
    if args.split == 'val' and args.duration != 5:
        print("Warning: Validation split typically uses 5min duration")
        print("         Using non-standard duration may affect comparability")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct expected output filename for display
    # Train split includes n_samples in filename, val/test do not
    class_name = f"ce{args.duration}min"
    suffix = "_rule" if args.rule_only else ""
    if args.split == 'train':
        expected_filename = f"{class_name}_{args.split}{suffix}_indices_{args.n_samples}.npz"
    else:
        expected_filename = f"{class_name}_{args.split}{suffix}_indices.npz"
    output_file = output_dir / expected_filename

    print("=" * 70)
    print("CE Index File Generator")
    print("=" * 70)
    print(f"Duration:       CE{args.duration}min")
    print(f"Split:          {args.split}")
    print(f"Test type:      {'Rule generalization only' if args.rule_only else 'Subject + temporal generalization' if args.split == 'test' else 'N/A'}")
    print(f"Samples:        {args.n_samples}")
    print(f"Positive only:  {args.positive_only}")
    print(f"Output:         {output_file}")
    print("=" * 70)

    # Get generator class
    GeneratorClass = DURATION_TO_CLASS[args.duration]

    # Map dataset split to multimodal data category
    # Train and val use train split (subjects 1-4) with different random CE combinations
    # Test uses test split (subject 5) for cross-subject + temporal generalization
    # Test with --rule-only uses train split (subjects 1-4) for CE pattern generalization only
    if args.rule_only:
        data_cat = 'train'  # Same subjects, tests CE pattern generalization
    else:
        data_cat = 'test' if args.split == 'test' else 'train'

    # Create generator instance
    print(f"\nInitializing CE{args.duration}min generator...")
    print(f"  Using multimodal data split: {data_cat}")
    print(f"  Window size: {WINDOW_SIZE}s")
    generator = GeneratorClass(n_data=args.n_samples, data_cat=data_cat, time_unit=WINDOW_SIZE)

    # Generate and save index file
    print(f"\nGenerating index dataset with {args.n_samples} samples...")
    rule_suffix = "_rule" if args.rule_only else ""
    generator.save_index_dataset(output_dir=str(output_dir), is_positive_sample=args.positive_only, suffix=rule_suffix, split_name=args.split)

    file_size_mb = os.path.getsize(output_file) / (1024**2)

    print("\n" + "=" * 70)
    print("✓ Index File Generated")
    print("=" * 70)
    print(f"Output:      {output_file}")
    print(f"Size:        {file_size_mb:.3f} MB")
    print(f"Samples:     {args.n_samples}")

if __name__ == '__main__':
    main()
