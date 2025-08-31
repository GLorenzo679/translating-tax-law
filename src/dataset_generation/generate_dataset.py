"""
Generate dataset from Catala source files.

This script processes .catala_fr files from the raw/ directory and extracts
law text paired with corresponding Catala code implementations.
"""

import json
import os
from typing import Dict

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "raw")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")

# Markdown code fence markers
CATALA_BLOCK_START = "```catala"
METADATA_BLOCK_START = "```catala-metadata"
BLOCK_END = "```"


def process_file(file_path: str) -> Dict[str, str]:
    """
    Process a single Catala file to extract law text and code blocks.

    The function reads a .catala_fr file and extracts:
    - Natural language law text
    - Catala code blocks
    - Optional metadata blocks

    Args:
        file_path: Path to the Catala file to process

    Returns:
        Dictionary containing extracted samples with structure:
            sample_1: {"input": law_text, "output": catala_code}
            sample_2: {"input": law_text, "output": catala_code}
            ...
            metadata: Optional metadata if present
    """
    data = {}
    sample_count = 1

    # State tracking
    law_text = ""
    code_text = ""
    is_in_catala_block = False
    is_in_metadata_block = False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Handle code fence markers
        if line == CATALA_BLOCK_START:
            is_in_catala_block = True
            continue

        if line == METADATA_BLOCK_START:
            is_in_metadata_block = True
            continue

        if line.startswith(BLOCK_END):
            # End of code block - save the accumulated content
            if code_text.strip():
                if is_in_catala_block:
                    # Save law text + code pair as a sample
                    data[f"sample_{sample_count}"] = {
                        "input": law_text.strip(),
                        "output": code_text.strip(),
                    }
                    sample_count += 1
                    law_text = ""
                    code_text = ""

                elif is_in_metadata_block:
                    # Append to metadata (may have multiple metadata blocks)
                    existing_metadata = data.get("metadata", "")
                    if existing_metadata:
                        data["metadata"] = existing_metadata + "\n\n" + code_text.strip()
                    else:
                        data["metadata"] = code_text.strip()
                    code_text = ""

            # Reset block state
            is_in_catala_block = False
            is_in_metadata_block = False
            continue

        # Accumulate text based on current context
        is_in_code_block = is_in_catala_block or is_in_metadata_block
        is_comment = line.startswith("#")

        if is_in_code_block and not is_comment:
            code_text += line + "\n"
        else:
            law_text += line + "\n"

    return data


def process_folder(folder: str) -> None:
    """
    Process all Catala files within a specific folder.

    Processes each .catala_fr file in the given folder and generates
    corresponding JSON files in the output directory.

    Args:
        folder: Name of the folder containing Catala files
    """
    folder_path = os.path.join(INPUT_PATH, folder)

    # Skip if not a directory
    if not os.path.isdir(folder_path):
        return

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".catala_fr"):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {folder}/{file_name}...")

        data = process_file(file_path)

        if data:
            output_folder = os.path.join(OUTPUT_PATH, folder)
            output_file = file_name.replace(".catala_fr", ".json")
            save_json(data, output_folder, output_file)


def save_json(data: Dict[str, str], output_folder: str, output_file: str) -> None:
    """
    Write processed data into a JSON file.

    Creates output directory if it doesn't exist and writes the processed
    data as a formatted JSON file.

    Args:
        data: Processed data to write
        output_folder: Path to output folder
        output_file: Name of the output JSON file
    """
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, output_file)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    """
    Main entry point for dataset generation.

    Walks through all folders in the input directory, processes each
    .catala_fr file, and generates corresponding JSON files in the output
    directory structure.
    """
    print(f"Reading Catala files from: {INPUT_PATH}")
    print(f"Writing JSON files to: {OUTPUT_PATH}\n")

    for folder in os.listdir(INPUT_PATH):
        process_folder(folder)

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
