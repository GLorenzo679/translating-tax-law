import zss
from tree_generator import TreeGenerator


# Define cost functions
def insert_cost(node):
    # print(f"\ninsert node: {node}\n")
    return 1


def remove_cost(node):
    # print(f"\nremove node: {node}\n")
    return 1


def update_cost(node1, node2):
    label1 = node1.label
    label2 = node2.label

    # print(f"\nnode1: {node1} \nnode2: {node2}\n")

    return 1 if label1 != label2 else 0
    # return 1 if node1 != node2 else 0


def compute(predicted_output_code: str, actual_output_code: str) -> float:

    # Add ```catala code ``` to the strings
    predicted_output_code = "```catala\n" + predicted_output_code + "\n```"
    actual_output_code = "```catala\n" + actual_output_code + "\n```"

    # Initialize the generator
    tree_gen = TreeGenerator()

    # Build the trees (tree-sitter)
    predicted_tree = tree_gen.build_tree_sitter(predicted_output_code)
    actual_tree = tree_gen.build_tree_sitter(actual_output_code)

    # # print trees just for showing purpose
    print("Actual tree:")
    tree_gen.print_tree(actual_tree.root_node)
    print(f"\nActual tree has ERRORS: {actual_tree.root_node.has_error}\n")
    # print("\n\n")
    # print("Predicted tree:")
    # tree_gen.print_tree(predicted_tree.root_node)
    # print(f"\nPredicted tree has ERRORS: {predicted_tree.root_node.has_error}\n")

    # Convert the trees to ZSS trees
    predicted_zss_tree, nodes_pred = tree_gen.convert_tree_sitter_in_zss_tree(
        predicted_tree.root_node
    )
    actual_zss_tree, nodes_act = tree_gen.convert_tree_sitter_in_zss_tree(
        actual_tree.root_node
    )

    # Compute the max number of nodes, removing the always present 4 nodes
    # (source_file, code_block, BEGIN_CODE, END_CODE)
    max_nodes = max(nodes_pred, nodes_act) - 4

    # Compute the tree distance
    distance = zss.distance(
        A=predicted_zss_tree,
        B=actual_zss_tree,
        get_children=zss.Node.get_children,
        insert_cost=insert_cost,
        remove_cost=remove_cost,
        update_cost=update_cost,
    )

    # print(f"\ndistance: {distance}")
    # print(f"number of nodes: {nodes_act} (act), {nodes_pred} (pred)")

    return distance / max_nodes


# Example usage
if __name__ == "__main__":

    actual_output_code = """
    champ d'application CalculAidePersonnaliséeLogementLocatif\n
    sous condition date_courante >= |2023-01-01| et date_courante < |2023-10-01|:\n
    exception métropole\n
    définition abattement_forfaitaire_d823_17\n
    sous condition\n(selon résidence sous forme\n
    -- Guadeloupe: vrai\n-- Martinique: vrai\n
    -- LaRéunion: vrai\n-- Mayotte: vrai\n
    -- SaintBarthélemy: vrai\n
    -- SaintMartin: vrai\n
    -- n'importe quel: faux) et\n
    nombre_personnes_à_charge = 1\nconséquence égal à\n
    8 181 €
    """

    predicted_output_code = """
    champ d'application CalculAidePersonnaliséeLogementLocatif
    sous condition date_courante >= |2023-01-01|:\n
    exception métropole\ndéfinition abattement égal à\n
    selon résidence sous forme\n
    -- Guadeloupe: vrai
    -- Guyane: vrai\n-- Martinique: vrai
    -- LaRéunion: vrai
    -- Mayotte: vrai
    -- SaintBarthélemy: vrai
    -- SaintMartin: vrai
    -- n'importe: faux
    exception base définition montant égal à
    si abattement alors
    8 181 €
    sinon
    0€
    """

    actual_output_code_diagram_draw = """
    champ d'application CalculAidePersonnaliséeLogementLocatif\n
    sous condition date_courante >= |2023-01-01| et date_courante < |2023-10-01|:\n
    exception métropole\n
    """
    predicted_output_code_diagram_draw = """
    champ d'application CalculAidePersonnaliséeLogementLocatif
    sous condition date_courante >= |2023-01-01|:\n
    """

    # actual_output_code = """
    # champ d'application CalculAidePersonnaliséeLogementLocatif\n
    # sous condition date_courante >= |2023-01-01|
    # """
    # predicted_output_code = """
    # champ d'application CalculAidePersonnaliséeLogementLocatif\n
    # sous condition date_courante > |2023-01-01|
    # """

    distance = compute(predicted_output_code_diagram_draw, actual_output_code_diagram_draw)
    print(f"\nTree distance: {distance:.3f}\n")
