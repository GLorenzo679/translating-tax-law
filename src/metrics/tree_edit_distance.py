from typing import List

import zss

from lib.tree_generator import TreeGenerator

def compute(predictions: List[str], references: List[str]) -> float:
    predictions = "```catala\n" + predictions[0] + "\n```"
    references = "```catala\n" + references[0] + "\n```"

    tree_gen = TreeGenerator()

    predicted_tree = tree_gen.build_tree_sitter(predictions)
    actual_tree = tree_gen.build_tree_sitter(references)

    predicted_zss_tree, nodes_pred = tree_gen.convert_tree_sitter_in_zss_tree(
        predicted_tree.root_node
    )
    actual_zss_tree, nodes_act = tree_gen.convert_tree_sitter_in_zss_tree(
        actual_tree.root_node
    )

    nodes_pred -= 4
    nodes_act -= 4
    max_nodes = max(nodes_pred, nodes_act)

    distance = zss.distance(
        A=predicted_zss_tree,
        B=actual_zss_tree,
        get_children=zss.Node.get_children,
        insert_cost=lambda node: 1,
        remove_cost=lambda node: 1,
        update_cost=lambda node1, node2: 1 if node1.label != node2.label else 0,
    )

    # ted stands for Tree Edit Distance
    normalized_ted = distance / max_nodes
    if normalized_ted > 1: # Should not happen
        print(f"Distance: {distance}")
        print(f"Nodes pred: {nodes_pred}, Nodes act: {nodes_act}")
        print("Actual tree:")
        tree_gen.print_tree(actual_tree.root_node)
        print("\n\n")
        print("Predicted tree:")
        tree_gen.print_tree(predicted_tree.root_node)
        # Clip res to 1
        normalized_ted = 1

    return normalized_ted
