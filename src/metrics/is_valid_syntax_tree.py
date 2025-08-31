from typing import List

from lib.tree_generator import TreeGenerator


def compute(predictions: List[str]) -> bool:
    predictions = "```catala\n" + predictions[0] + "\n```"

    tree_gen = TreeGenerator()
    is_valid = True

    predicted_tree = tree_gen.build_tree_sitter(predictions)
    predicted_tree_string = tree_gen.tree_to_string(predicted_tree.root_node)

    if any(word in predicted_tree_string.lower() for word in ["err", "error"]):
        is_valid = False

    return is_valid
