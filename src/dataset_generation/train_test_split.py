"""
Split dataset into training and test sets.

This script randomly shuffles and splits the combined dataset into
train.json and test.json files for model training and evaluation.
"""

import json
import os
import random
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")

# Split configuration
DEFAULT_TRAIN_RATIO = 0.85  # 85% train, 15% test
RANDOM_SEED = 42  # For reproducibility


def split_dataset(
    samples: List[Dict], train_ratio: float = DEFAULT_TRAIN_RATIO
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and test sets with random shuffling.

    Args:
        samples: List of samples from dataset
        train_ratio: Fraction of data to use for training (default: 0.85)

    Returns:
        Tuple of (train_samples, test_samples)
    """
    # Set seed for reproducible splits
    random.seed(RANDOM_SEED)

    # Shuffle samples randomly
    shuffled = samples.copy()
    random.shuffle(shuffled)

    # Calculate split index
    split_idx = int(len(shuffled) * train_ratio)

    # Split into train/test
    train_samples = shuffled[:split_idx]
    test_samples = shuffled[split_idx:]

    return train_samples, test_samples


def save_split(samples: List[Dict], file_path: str) -> None:
    """
    Save a dataset split to a JSON file.

    Args:
        samples: List of samples to save
        file_path: Path to output file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def main():
    """
    Main entry point for dataset splitting.

    Reads dataset.json, splits it into train/test sets, and saves them
    to separate files.
    """
    print(f"Reading dataset from: {INPUT_PATH}/dataset.json\n")

    # Read dataset
    dataset_path = os.path.join(INPUT_PATH, "dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Split dataset
    train_samples, test_samples = split_dataset(samples)

    # Print split statistics
    print(f"Dataset split (seed={RANDOM_SEED}, ratio={DEFAULT_TRAIN_RATIO}):")
    print(f"  Total samples: {len(samples)}")
    print(f"  Train samples: {len(train_samples)} ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"  Test samples:  {len(test_samples)} ({len(test_samples)/len(samples)*100:.1f}%)")

    # Save splits
    train_path = os.path.join(INPUT_PATH, "train.json")
    test_path = os.path.join(INPUT_PATH, "test.json")

    save_split(train_samples, train_path)
    save_split(test_samples, test_path)

    print(f"\nSaved train set to: {train_path}")
    print(f"Saved test set to: {test_path}")


if __name__ == "__main__":
    main()
