"""
Transliterate the JA homograph Excel dataset from Hebrew script to Arabic script.
Based on the transliteration flow proposed by Moreno Gonzalez et al. (2025): https://arxiv.org/abs/2507.04746

Produces two output files (dotted and dotless) that can be used directly
with the existing homograph clustering pipeline (pipeline.py) -- no pipeline
changes needed.

Usage:
    python arabic/transliterate_homograph_dataset.py
"""

import sys
import io
import re
from pathlib import Path

import pandas as pd



# Ensure stdout can handle Arabic/Hebrew characters on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Allow running from repo root or from arabic/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "arabic"))

from transliterate_ja2arabic import transliterate_ja_to_arabic


# INPUT_XLSX = REPO_ROOT / "homograph_analysis" / "homograph_data" / "JA_Homograph_DB_human_renumbered_final.xlsx"
INPUT_XLSX = REPO_ROOT / "homograph_analysis" / "homograph_data" / "JA_Homograph_after_submitted.xlsx"
OUTPUT_DIR = REPO_ROOT / "homograph_analysis" / "homograph_data"

SHEET_NAME = "samples"


def clean_sample_for_word_location(text):
    """
    Mirror the cleaning logic from homograph_dataset_processing.py:
    strip commas, periods, and apostrophes so that word lookup succeeds.
    """
    to_remove = ",.':[]{}"
    for c in to_remove:
        text = text.replace(c, '')
    return text

def compute_word_index(sample, word, instance_id):
    """Find the 1-indexed occurrence of `word` in sample.split(), return its position."""
    words = sample.split()
    occurrence_count = 0
    for i, w in enumerate(words):
        if w == word:
            occurrence_count += 1
            if occurrence_count == instance_id:
                return i
    return -1


def transliterate_dataset(input_xlsx, output_xlsx, use_dots):
    """
    Read the original Hebrew-script homograph Excel, transliterate sample and
    word columns to Arabic, validate word positions, and write the result.
    """
    df = pd.read_excel(input_xlsx, sheet_name=SHEET_NAME)
    print(f"Loaded {len(df)} rows from {input_xlsx}")

    mode_label = "dotted" if use_dots else "dotless"
    warning_count = 0

    for idx in df.index:
        original_sample = str(df.at[idx, 'sample'])
        original_word = str(df.at[idx, 'word'])
        instance_id = int(df.at[idx, 'instance_id'])

        arabic_sample = transliterate_ja_to_arabic(original_sample, use_dots=use_dots)
        arabic_word = transliterate_ja_to_arabic(original_word, use_dots=use_dots)

        cleaned = clean_sample_for_word_location(arabic_sample)
        word_index = compute_word_index(cleaned, arabic_word, instance_id)

        if word_index == -1:
            warning_count += 1
            row_id = df.at[idx, 'id'] if 'id' in df.columns else idx
            print(
                f"  WARNING [{mode_label}] row id={row_id}: "
                f"transliterated word '{arabic_word}' not found at instance {instance_id} "
                f"in cleaned sample: '{cleaned}'"
            )

        df.at[idx, 'sample'] = arabic_sample
        df.at[idx, 'word'] = arabic_word

    print(f"Transliteration complete ({mode_label}). Warnings: {warning_count}/{len(df)}")
    df.to_excel(output_xlsx, sheet_name=SHEET_NAME, index=False)
    print(f"Written to {output_xlsx}")
    return warning_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dotted_path = OUTPUT_DIR / "Arabic_Homograph_dotted.xlsx"
    dotless_path = OUTPUT_DIR / "Arabic_Homograph_dotless.xlsx"

    print("=== Dotted transliteration ===")
    transliterate_dataset(INPUT_XLSX, dotted_path, use_dots=True)

    print()
    print("=== Dotless transliteration ===")
    transliterate_dataset(INPUT_XLSX, dotless_path, use_dots=False)


if __name__ == "__main__":
    main()
