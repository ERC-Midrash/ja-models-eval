import json
import re

import unicodedata
from typing import List, Dict, Any
from pathlib import Path


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all entries from a JSONL file."""
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def filter_dataset(entries: List[Dict[str, Any]], word_id: int, sentence_ids:List=None) -> List[Dict[str, Any]]:
    """Filter entries to only those with the given word_id and optionally specific sentence ids."""
    entries = [e for e in entries if e.get("word_id") == word_id]
    if sentence_ids is not None:
        entries = [e for e in entries if e.get("sentence_id") in sentence_ids]
    return entries

def get_embedding_path(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    output_path = Path(config['outfolder']) / f"embeddings_{config['model'].lower()}_layer{abs(int(config['layer']))}.jsonl"
    return output_path


def remove_diacritics(dia_str):
    # Normalize the string to Unicode NFC form
    normalized_string = unicodedata.normalize('NFC', dia_str)

    # Dictionary mapping Hebrew characters with dagesh to their base forms
    replacements = {
        # Hebrew letters with dagesh mapped to their base forms
        '\u05D1\u05BC': '\u05D1',  # bet with dagesh -> bet
        '\u05D2\u05BC': '\u05D2',  # gimel with dagesh -> gimel
        '\u05D3\u05BC': '\u05D3',  # dalet with dagesh -> dalet
        '\u05D4\u05BC': '\u05D4',  # he with dagesh -> he
        '\u05D5\u05BC': '\u05D5',  # vav with dagesh -> vav
        '\u05D6\u05BC': '\u05D6',  # zayin with dagesh -> zayin
        '\u05D8\u05BC': '\u05D8',  # tet with dagesh -> tet
        '\u05D9\u05BC': '\u05D9',  # yod with dagesh -> yod
        '\u05DB\u05BC': '\u05DB',  # kaf with dagesh -> kaf
        '\u05DC\u05BC': '\u05DC',  # lamed with dagesh -> lamed
        '\u05DE\u05BC': '\u05DE',  # mem with dagesh -> mem
        '\u05E0\u05BC': '\u05E0',  # nun with dagesh -> nun
        '\u05E1\u05BC': '\u05E1',  # samekh with dagesh -> samekh
        '\u05E4\u05BC': '\u05E4',  # pe with dagesh -> pe
        '\u05E6\u05BC': '\u05E6',  # tsadi with dagesh -> tsadi
        '\u05E7\u05BC': '\u05E7',  # qof with dagesh -> qof
        '\u05E9\u05BC': '\u05E9',  # shin with dagesh -> shin
        '\u05EA\u05BC': '\u05EA',  # tav with dagesh -> tav

        # Precomposed characters (FB30-FB4F range)
        '\uFB31': '\u05D1',  # bet with dagesh
        '\uFB32': '\u05D2',  # gimel with dagesh
        '\uFB33': '\u05D3',  # dalet with dagesh
        '\uFB34': '\u05D4',  # he with dagesh
        '\uFB35': '\u05D5',  # vav with dagesh
        '\uFB4B': '\u05D5',  # vav with holam
        '\uFB36': '\u05D6',  # zayin with dagesh
        '\uFB38': '\u05D8',  # tet with dagesh
        '\uFB39': '\u05D9',  # yod with dagesh
        '\uFB3A': '\u05DA',  # final kaf with dagesh
        '\uFB3B': '\u05DB',  # kaf with dagesh
        '\uFB3C': '\u05DC',  # lamed with dagesh
        '\uFB3E': '\u05DE',  # mem with dagesh
        '\uFB40': '\u05E0',  # nun with dagesh
        '\uFB41': '\u05E1',  # samekh with dagesh
        '\uFB43': '\u05E3',  # final pe with dagesh
        '\uFB44': '\u05E4',  # pe with dagesh
        '\uFB46': '\u05E6',  # tsadi with dagesh
        '\uFB47': '\u05E7',  # qof with dagesh
        '\uFB48': '\u05E8',  # resh with dagesh
        '\uFB2A': '\u05E9',  # shin with shin dot
        '\uFB2B': '\u05E9',  # shin with sin dot
        '\uFB2C': '\u05E9',  # shin with dagesh and shin dot
        '\uFB2D': '\u05E9',  # shin with dagesh and sin dot
        '\uFB49': '\u05E9',  # shin with dagesh
        '\uFB4A': '\u05EA',  # tav with dagesh
    }

    # First, replace precomposed characters
    result = ''
    i = 0
    while i < len(normalized_string):
        char = normalized_string[i]
        if char in replacements:
            result += replacements[char]
        else:
            result += char
        i += 1

    # Now handle combining diacritics
    result_with_dots = ''
    for char in result:
        if char == '\u0307':  # Dot above (U+0307)
            result_with_dots += "'"
        elif not unicodedata.combining(char):
            result_with_dots += char

    # Remove remaining diacritics
    final_result = ''.join(c for c in result_with_dots if not unicodedata.combining(c))

    return final_result


def remove_punctuation(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected string in 'sample' column, got {type(text).__name__}: {text!r}. Check your data for empty/NaN values.")
    return text.replace(',', '').replace('.', '')


def remove_geresh(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected string in 'sample' column, got {type(text).__name__}: {text!r}. Check your data for empty/NaN values.")
    return text.replace("'", "")


def clean_paren(text):
    text = text.replace('(?)', '')
    text = re.sub(r'\[\.+\]', 'GGAAPP', text)
    for c in ['{', '}','[', ']']:
        text = text.replace(c, '')
    text = text.replace("GGAAPP", "[GAP]")
    return text
