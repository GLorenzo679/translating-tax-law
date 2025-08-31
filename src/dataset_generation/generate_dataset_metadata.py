"""
Generate metadata from Catala construct definitions.

This script extracts metadata (enumerations, structures, and scopes) from
JSON files and combines them into a single metadata.json file for reference.
"""

import json
import os
from typing import Dict, Optional

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")

# Catala construct declaration keywords
DECLARATION_TYPES = [
    "déclaration énumération",      # enumeration declaration
    "déclaration structure",         # structure declaration
    "déclaration champ d'application",  # scope declaration
]


def extract_construct_name(declaration_line: str) -> Optional[str]:
    """
    Extract the name of a construct from its declaration line.

    Handles three types of Catala declarations:
    - Enumerations (énumération)
    - Structures (structure)
    - Scopes (champ d'application)

    Args:
        declaration_line: Line containing the construct declaration

    Returns:
        Name of the construct (without trailing colon), or None if not a valid declaration
    """
    for declaration_keyword in DECLARATION_TYPES:
        if declaration_line.startswith(declaration_keyword):
            words = declaration_line.split()
            # Extract name and remove trailing colon
            # Format: "déclaration <type> <name>:" or "déclaration <type> d'<name>:"
            name_index = 2 if len(words) == 3 else 3
            return words[name_index].rstrip(":")

    return None


def parse_metadata_content(metadata_text: str, constructs: Dict[str, str]) -> Dict[str, str]:
    """
    Parse metadata text and extract construct definitions.

    Constructs are separated by blank lines. Each construct starts with a
    declaration line and continues until a blank line is encountered.

    Args:
        metadata_text: Raw metadata text containing construct definitions
        constructs: Dictionary to accumulate construct definitions

    Returns:
        Dictionary mapping construct names to their full definitions
    """
    current_construct_name = None

    for line in metadata_text.split("\n"):
        # Check if this line starts a new construct declaration
        if not current_construct_name:
            current_construct_name = extract_construct_name(line)
            if current_construct_name:
                # Start a new construct or append to existing one
                if current_construct_name in constructs:
                    print(f"Warning: Duplicate construct '{current_construct_name}' - appending")
                    constructs[current_construct_name] += line + "\n"
                else:
                    constructs[current_construct_name] = line + "\n"
            continue

        # Empty line marks end of construct definition
        if not line.strip():
            current_construct_name = None
        else:
            # Continue accumulating lines for current construct
            constructs[current_construct_name] += line + "\n"

    return constructs


def process_file(file_path: str, constructs: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Process a single JSON file to extract metadata definitions.

    Args:
        file_path: Path to the JSON file
        constructs: Dictionary to accumulate metadata definitions

    Returns:
        Updated constructs dictionary if file contains metadata, None otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metadata" not in data:
        return None

    parse_metadata_content(data["metadata"], constructs)
    return constructs


def save_json(data: Dict[str, str], output_folder: str, output_file: str) -> None:
    """
    Write data to a JSON file.

    Args:
        data: Data to write
        output_folder: Folder path for output
        output_file: Name of the output file
    """
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, output_file)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    """
    Main entry point for metadata generation.

    Scans all JSON files in the dataset directory, extracts construct definitions
    from metadata blocks, and combines them into a single metadata.json file.
    """
    print(f"Reading JSON files from: {INPUT_PATH}")
    print(f"Writing metadata to: {OUTPUT_PATH}/metadata.json\n")

    constructs = {}

    for folder in os.listdir(INPUT_PATH):
        folder_path = os.path.join(INPUT_PATH, folder)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {folder}/{file_name}...")

            process_file(file_path, constructs)

    # Sort constructs alphabetically for easier reference
    constructs = dict(sorted(constructs.items()))

    save_json(constructs, OUTPUT_PATH, "metadata.json")
    print(f"\nExtracted {len(constructs)} construct definitions")


if __name__ == "__main__":
    main()
