"""
Collect and combine dataset samples into a single file.

This script aggregates all individual sample JSON files into a single
dataset.json file suitable for training.
"""

import json
import os
from typing import List, Dict, Any

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")


def format_metadata(metadata: Dict[str, str]) -> str:
    """
    Format metadata dictionary into a string.

    Concatenates all construct definitions with newlines between them.

    Args:
        metadata: Dictionary of construct definitions {name: definition}

    Returns:
        Formatted metadata string with definitions separated by newlines
    """
    if not metadata:
        return ""

    return "\n".join(metadata.values())


def collect_samples_from_file(file_path: str) -> List[Dict[str, str]]:
    """
    Extract all samples from a single JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of samples, each with input, metadata, and output fields
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for sample_id, sample in data.items():
        # Skip file-level metadata entry
        if sample_id == "metadata":
            continue

        input_text = sample.get("input", "")
        metadata_dict = sample.get("metadata", {})
        output_text = sample.get("output", "")

        samples.append({
            "input": input_text,
            "metadata": format_metadata(metadata_dict),
            "output": output_text,
        })

    return samples


def main():
    """
    Main entry point for dataset collection.

    Scans all JSON files in the dataset directory, extracts samples, and
    combines them into a single dataset.json file.

    Output format:
    [
        {
            "input": "law text...",
            "metadata": "construct definitions...",
            "output": "catala code..."
        },
        ...
    ]
    """
    print(f"Collecting samples from: {INPUT_PATH}")
    print(f"Writing combined dataset to: {OUTPUT_PATH}/dataset.json\n")

    dataset = []

    for folder in os.listdir(INPUT_PATH):
        folder_path = os.path.join(INPUT_PATH, folder)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".json") or file_name == "metadata.json":
                continue

            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {folder}/{file_name}...")

            samples = collect_samples_from_file(file_path)
            dataset.extend(samples)

    output_file = os.path.join(OUTPUT_PATH, "dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nCollected {len(dataset)} samples into dataset.json")


if __name__ == "__main__":
    main()
