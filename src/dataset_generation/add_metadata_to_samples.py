"""
Add metadata to dataset samples.

This script identifies which construct definitions (from metadata.json) are
referenced in each sample's output code and adds them to the sample.
"""

import json
import os
import re
from typing import Dict, Set

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")


def build_reference_pattern(name: str) -> str:
    """
    Build a regex pattern to match references to a construct.

    Matches:
    - Direct references: `construct_name`
    - Field access: `construct_name.field`
    - Qualified names: `module.construct_name`

    Args:
        name: Name of the construct to match

    Returns:
        Regex pattern string
    """
    return (
        rf"\b{name}\b|"  # Direct reference
        rf"\b{name}\.[A-Za-z][A-Za-z0-9_]*\b|"  # Field access
        rf"\b[A-Za-z][A-Za-z0-9_]*\.{name}\b"  # Qualified name
    )


def find_nested_constructs(
    definition: str, all_constructs: Dict[str, str], visited: Set[str]
) -> Dict[str, str]:
    """
    Recursively find constructs referenced within a definition.

    This handles cases where one construct references another, which in turn
    references more constructs (transitive dependencies).

    Args:
        definition: The definition text to scan
        all_constructs: Dictionary of all available construct definitions
        visited: Set of already processed definitions (prevents infinite loops)

    Returns:
        Dictionary of nested constructs and their definitions
    """
    if definition in visited:
        return {}

    visited.add(definition)
    nested = {}

    for construct_name, construct_def in all_constructs.items():
        pattern = build_reference_pattern(construct_name)
        if re.search(pattern, definition, re.IGNORECASE):
            nested[construct_name] = construct_def
            # Recursively find constructs referenced in this construct
            nested.update(find_nested_constructs(construct_def, all_constructs, visited))

    return nested


def find_relevant_constructs(
    output_text: str, all_constructs: Dict[str, str]
) -> Dict[str, str]:
    """
    Find all constructs referenced in the output text (including nested references).

    Scans the output code for construct references and recursively includes
    any constructs that those constructs depend on.

    Args:
        output_text: The Catala code to scan for construct references
        all_constructs: Dictionary of all available construct definitions

    Returns:
        Dictionary of relevant constructs and their definitions
    """
    relevant = {}
    visited = set()

    for construct_name, definition in all_constructs.items():
        pattern = build_reference_pattern(construct_name)

        if re.search(pattern, output_text, re.IGNORECASE):
            # Found a reference to this construct
            relevant[construct_name] = definition
            # Find any constructs this construct depends on
            relevant.update(find_nested_constructs(definition, all_constructs, visited))

    return relevant


def process_file(file_path: str, all_constructs: Dict[str, str]) -> bool:
    """
    Process a single JSON file to add metadata to each sample.

    Args:
        file_path: Path to the JSON file to process
        all_constructs: Dictionary of all construct definitions

    Returns:
        True if file was modified, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False

    for sample_id, sample in data.items():
        # Skip the file-level metadata entry
        if sample_id == "metadata":
            continue

        output_text = sample.get("output", "")
        relevant_constructs = find_relevant_constructs(output_text, all_constructs)

        if relevant_constructs:
            sample["metadata"] = relevant_constructs
            modified = True

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return modified


def main():
    """
    Main entry point for adding metadata to samples.

    Loads the global metadata.json file and adds relevant construct definitions
    to each sample based on which constructs are referenced in the sample's code.
    """
    print(f"Reading metadata from: {INPUT_PATH}/metadata.json")
    print(f"Processing samples in: {INPUT_PATH}\n")

    metadata_path = os.path.join(INPUT_PATH, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_constructs = json.load(f)

    print(f"Loaded {len(all_constructs)} construct definitions\n")

    samples_modified = 0

    for folder in os.listdir(INPUT_PATH):
        folder_path = os.path.join(INPUT_PATH, folder)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".json") or file_name == "metadata.json":
                continue

            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {folder}/{file_name}...")

            if process_file(file_path, all_constructs):
                samples_modified += 1

    print(f"\nModified {samples_modified} files with metadata")


if __name__ == "__main__":
    main()
