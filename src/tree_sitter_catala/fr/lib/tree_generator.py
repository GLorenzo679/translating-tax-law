import ctypes
import os

from tree_sitter import Language, Parser
from zss import Node

# class Node:
#     def __init__(self, label):
#         self.label = label
#         self.children = []

#     def add_child(self, child):
#         self.children.append(child)

#     def __str__(self):
#         return self.label

#     def __repr__(self):
#         return f"Node({self.label})"


class TreeGenerator:
    def __init__(self, lib_path="./../catala.so", language_name="catala_fr"):
        """
        Initialize the Language and Parser for Tree-sitter.

        :param lib_path: Path to the shared library (e.g., 'catala.so').
        :param language_name: Language name as defined in the Tree-sitter grammar.
        """
        # Load the shared library using ctypes
        lib = ctypes.CDLL(os.path.abspath(lib_path))

        # Get the language function and specify the return type
        tree_sitter_lang_func = getattr(lib, f"tree_sitter_{language_name}")
        tree_sitter_lang_func.restype = ctypes.POINTER(ctypes.c_void_p)

        # Get the language pointer and initialize the Language object
        lang_ptr = tree_sitter_lang_func()
        lang_ptr_int = ctypes.cast(lang_ptr, ctypes.c_void_p).value
        self.language = Language(lang_ptr_int)

        # Initialize the parser
        self.parser = Parser(self.language)

    def build_tree_sitter(self, code):
        """
        Build a syntax tree from a code string.

        :param code: Source code as a string.
        :return: Tree-sitter syntax tree object.
        """
        if isinstance(code, str):
            code = code.encode("utf-8")  # Convert to bytes if necessary
        return self.parser.parse(code)

    @staticmethod
    def convert_tree_sitter_in_zss_tree(node):
        """
        Convert a Tree-sitter syntax tree to a ZSS tree.

        :param node: The root node of the Tree-sitter tree.
        :return: The root node of the ZSS tree.
        """
        root = Node(node.type)
        number_of_nodes = 1
        for child in node.children:
            child_node, child_nodes = TreeGenerator.convert_tree_sitter_in_zss_tree(
                child
            )
            root.addkid(child_node)
            number_of_nodes += child_nodes
        return root, number_of_nodes

    @staticmethod
    def print_tree(node, indent=""):
        """
        Print the syntax tree in a readable format.

        :param node: The root node of the tree.
        :param indent: Indentation string for formatting.
        """
        print(f"{indent}{node.type}")
        for child in node.children:
            TreeGenerator.print_tree(child, indent + "  ")


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    # lib_path = "catala.so"  # Path to your shared library
    # language_name = "catala_fr"  # Language name as defined in the grammar

    tree_gen = TreeGenerator()

    # Example code
    code = """
    ```catala
    champ d'application InterfaceAllocationsFamiliales:
    définition enfants_à_charge égal à
    (Enfant {
    -- identifiant : enfant.d_identifiant
    -- rémuneration_mensuelle : enfant.d_rémuneration_mensuelle
    -- date_de_naissance : enfant.d_date_de_naissance
    -- prise_en_charge : enfant.d_prise_en_charge
    -- obligation_scolaire :
    (si enfant.d_date_de_naissance + 3 an >= i_date_courante alors
    SituationObligationScolaire.Avant
    sinon (si enfant.d_date_de_naissance + 16 an >= i_date_courante allora
    SituationObligationScolaire.Pendant
    sinon SituationObligationScolaire.Après))
    -- a_déjà_ouvert_droit_aux_allocations_familiales:
    enfant.d_a_déjà_ouvert_droit_aux_allocations_familiales
    -- bénéficie_titre_personnel_aide_personnelle_logement:
    enfant.d_bénéficie_titre_personnel_aide_personnelle_logement
    }
    pour enfant parmi i_enfants)
   ```
    """

    # Build the tree
    tree = tree_gen.build_tree_sitter(code)

    # Print the tree
    tree_gen.print_tree(tree.root_node)
