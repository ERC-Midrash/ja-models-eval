"""
Apply SWEET post-correction to the transliterated Arabic homograph dataset.
Based on the SWEET correction flow proposed by Moreno Gonzalez et al. (2025): https://arxiv.org/abs/2507.04746

Reads Arabic_Homograph_dotted.xlsx (produced by transliterate_homograph_dataset.py),
runs each sample through the SWEET model to fix transliteration errors, then writes
Arabic_Homograph_sweet_dotted.xlsx with corrected samples and words.

Word alignment strategy:
  - If SWEET preserves word count, extract the corrected word at the same index.
  - If word count changed, try to find the original word in the corrected text.
  - If that fails, fall back to the uncorrected sample and word.

Usage:
    python arabic/sweet_correct_homograph_dataset.py
"""

import sys
import io
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arabic.sweet_rewrite import load_sweet_model, predict


INPUT_XLSX = REPO_ROOT / "homograph_analysis" / "homograph_data" / "Arabic_Homograph_dotted.xlsx"
OUTPUT_XLSX = REPO_ROOT / "homograph_analysis" / "homograph_data" / "Arabic_Homograph_sweet_dotted.xlsx"
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


def correct_dataset(input_xlsx, output_xlsx, decode_iter=2):
    """
    Apply SWEET post-correction to every sample in the dataset.

    For each row:
      1. Get the word's index in the original cleaned sample.
      2. Run SWEET on the full sample.
      3. If word count is preserved, use the corrected word at the same index.
      4. If word count changed, try to find the original word in the corrected sample.
      5. If not found, fall back to the uncorrected sample and word.
    """
    df = pd.read_excel(input_xlsx, sheet_name=SHEET_NAME)
    print(f"Loaded {len(df)} rows from {input_xlsx}")

    print("Loading SWEET model...")
    model, tokenizer = load_sweet_model()
    print("Model loaded.")

    corrected_count = 0
    preserved_count = 0
    fallback_word_found_count = 0
    fallback_uncorrected_count = 0
    no_change_count = 0

    df['sweet_status'] = 'IGNORE'  # IGNORE by default

    for idx in df.index:
        original_sample = str(df.at[idx, 'sample'])
        original_word = str(df.at[idx, 'word'])
        instance_id = int(df.at[idx, 'instance_id'])

        original_clean = clean_sample_for_word_location(original_sample)
        word_index = compute_word_index(original_clean, original_word, instance_id)

        words = original_clean.split()
        if len(words) == 0:
            continue

        corrected_sample = predict(model, tokenizer, words, decode_iter=decode_iter)

        if corrected_sample == original_clean:
            no_change_count += 1
            df.at[idx, 'sweet_status'] = "UNCHANGED"
            continue

        corrected_count += 1

        corrected_clean = clean_sample_for_word_location(corrected_sample)
        original_words = original_clean.split()
        corrected_words = corrected_clean.split()

        if len(original_words) == len(corrected_words) and corrected_words[word_index] == original_word:
            df.at[idx, 'sample'] = corrected_sample
            df.at[idx, 'word'] = original_word
            df.at[idx, 'sweet_status'] = "HOMOG_PRESERVED"
            preserved_count += 1

        else:
            corrected_word_index = compute_word_index(corrected_clean, original_word, instance_id)

            if corrected_word_index != -1:
                df.at[idx, 'sample'] = corrected_sample
                df.at[idx, 'sweet_status'] = "HOMOG_MOVED"
                fallback_word_found_count += 1
            else:
                fallback_uncorrected_count += 1
                row_id = df.at[idx, 'id'] if 'id' in df.columns else idx
                print(
                    f"  FALLBACK row id={row_id} (word count "
                    f"{len(original_words)} -> {len(corrected_words)}):"
                    f"word '{original_word}' not found in corrected sample. "
                    f"Keeping original."
                )
                try:
                    corrected_words[word_index] = f"=={corrected_words[word_index]}=="
                except:
                    pass
                print(f"\tSWEET output: {' '.join(corrected_words)}\n")

    df.to_excel(output_xlsx, sheet_name=SHEET_NAME, index=False)
    print(f"\nWritten to {output_xlsx}")
    print(f"\n=== Summary ===")
    print(f"  Total rows:                {len(df)}")
    print(f"  No change by SWEET:        {no_change_count}")
    print(f"  Corrected (word preserved): {preserved_count}")
    print(f"  Corrected (word found):     {fallback_word_found_count}")
    print(f"  Fallback to uncorrected:    {fallback_uncorrected_count}")
    print(f"  Total corrected:            {corrected_count}")


def main():
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    correct_dataset(INPUT_XLSX, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
