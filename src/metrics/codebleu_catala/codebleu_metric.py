import re
import sys
from pathlib import Path
from typing import List

from src.metrics.codebleu_catala.utils_codebleu import calc_codebleu_catala

src_path = str(Path(__file__).parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)


french_catala_keywords = [
    "alors",
    "argent",
    "assertion",
    "avec",
    "booléen",
    "champ d'application",
    "combinaison de",
    "condition",
    "conséquence non rempli",
    "conséquence rempli",
    "contenu",
    "contexte",
    "contient",
    "dans",
    "date",
    "date arrondi croissant",
    "date arrondi décroissant",
    "de",
    "donnée",
    "durée",
    "décimal",
    "déclaration",
    "dépend de",
    "entier",
    "entrée",
    "est maximum",
    "est minimum",
    "et",
    "exception",
    "existe",
    "faux",
    "initialement",
    "interne",
    "liste",
    "liste de",
    "mais en remplaçant",
    "maximum",
    "minimum",
    "n'importe quel",
    "nombre",
    "non",
    "on a",
    "ou",
    "ou bien",
    "parmi",
    "pour",
    "pour tout",
    "résultat",
    "selon",
    "si",
    "sinon",
    "soit",
    "somme",
    "sous condition",
    "sous forme",
    "structure",
    "tel que",
    "vrai",
    "égal à",
    "énumération",
    "état",
    "étiquette",
]


def compute(predictions: List[str], references: List[str]) -> dict:
    # Given that some keywords in Catala are composed of multiple words, we need to use a custom tokenizer to deal with them.
    # This tokenizer will split the input string into tokens, where each token is either a keyword or a sequence of non-whitespace characters.
    # A simple x.split() would not work because it would split the keywords into multiple tokens.
    def tokenizer_with_keywords(s, keywords=french_catala_keywords):
        keyword_pattern = re.compile(
            r"\b("
            + r"|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
            + r")\b"
        )
        tokens = []
        pos = 0
        while pos < len(s):
            match = keyword_pattern.match(s, pos)
            if match:
                tokens.append(match.group(0))
                pos = match.end()
            else:
                non_keyword_match = re.match(r"\S+", s[pos:])
                if non_keyword_match:
                    tokens.append(non_keyword_match.group(0))
                    pos += len(non_keyword_match.group(0))
                else:
                    pos += 1
        return tokens

    # The following metrics are available in the calc_codebleu_catala return dict:
    # codebleu
    # ngram_match_score
    # weighted_ngram_match_score
    # syntax_match_score
    # dataflow_match_score

    # alpha, beta, gamma and theta values for the codebleu calculation
    weights = (1 / 3, 1 / 3, 1 / 3, 0)
    return calc_codebleu_catala(
        references,
        predictions,
        weights=weights,
        keywords=french_catala_keywords,
        tokenizer=tokenizer_with_keywords,
    )
