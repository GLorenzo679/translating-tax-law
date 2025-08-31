import ctypes
import os

from tree_sitter import Language, Parser
from zss import Node


class TreeGenerator:
    def __init__(self, lib_path="lib/catala.so", language_name="catala_fr"):
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
        if node.type in ["variable"] or ("name" in node.type):
            print(f"{indent}{node.type} (actual name: {node.text.decode()})")
        else:
            print(f"{indent}{node.type}")
        for child in node.children:
            TreeGenerator.print_tree(child, indent + "  ")

    @staticmethod
    def tree_to_string(node, indent=""):
        """
        Convert the syntax tree into a string representation.

        :param node: The root node of the tree.
        :param indent: Indentation string for formatting.
        :return: A string representing the tree.
        """
        result = f"{indent}{node.type}\n"
        for child in node.children:
            result += TreeGenerator.tree_to_string(child, indent + "  ")
        return result
