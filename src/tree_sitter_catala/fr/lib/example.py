import ctypes
import os

from tree_sitter import Language, Parser


# Function to print the AST
def print_tree(node, indent=""):
    print(f"{indent}{node.type}")
    for child in node.children:
        print_tree(child, indent + "  ")


def main():
    # Define the path to the shared library
    lib_path = os.path.abspath("catala.so")

    # Load the shared library using ctypes
    lib = ctypes.CDLL(lib_path)

    # Get the language function and specify the return type
    tree_sitter_catala = lib.tree_sitter_catala_fr
    tree_sitter_catala.restype = ctypes.POINTER(ctypes.c_void_p)

    # Get the language pointer
    lang_ptr = tree_sitter_catala()

    # Convert the pointer to an integer
    lang_ptr_int = ctypes.cast(lang_ptr, ctypes.c_void_p).value

    # Initialize the Language object with the integer pointer
    CATALA_LANGUAGE = Language(lang_ptr_int)

    # Create a parser
    parser = Parser(CATALA_LANGUAGE)

    # Example code to parse
    # code = """
    # ```catala
    # champ d'application AllocationsFamiliales :
    #   exception
    #   définition plafond_I_d521_3 sous condition
    #     date_courante >= |2018-01-01| et date_courante <= |2018-12-31|
    #   conséquence égal à 56 286 € +
    #     5 628 € * (décimal de
    #       (nombre de enfants_à_charge_droit_ouvert_prestation_familiale))

    #   exception
    #   définition plafond_II_d521_3 sous condition
    #     date_courante >= |2018-01-01| et date_courante <= |2018-12-31|
    #   conséquence égal à 78 770 € +
    #     5 628 € * (décimal de
    #       (nombre de enfants_à_charge_droit_ouvert_prestation_familiale))
    # ```
    # """

    code = """
    ```catala
    champ d'application InterfaceAllocationsFamiliales:\ndéfinition enfants_à_charge égal à\n(Enfant {\n-- identifiant : enfant.d_identifiant\n-- rémuneration_mensuelle : enfant.d_rémuneration_mensuelle\n-- date_de_naissance : enfant.d_date_de_naissance\n-- prise_en_charge : enfant.d_prise_en_charge\n-- obligation_scolaire :\n(si enfant.d_date_de_naissance + 3 an >= i_date_courante alors\nSituationObligationScolaire.Avant\nsinon (si enfant.d_date_de_naissance + 16 an >= i_date_courante alors\nSituationObligationScolaire.Pendant\nsinon SituationObligationScolaire.Après))\n-- a_déjà_ouvert_droit_aux_allocations_familiales:\nenfant.d_a_déjà_ouvert_droit_aux_allocations_familiales\n-- bénéficie_titre_personnel_aide_personnelle_logement:\nenfant.d_bénéficie_titre_personnel_aide_personnelle_logement\n}\npour enfant parmi i_enfants)
    ```
    """

    code = bytes(code, "utf-8")

    # Parse the code
    tree = parser.parse(code)

    print()
    print_tree(tree.root_node)


if __name__ == "__main__":
    main()
