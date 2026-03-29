"""
Convert XLSX with homograph info into JSONL format
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from homograph_analysis.utils import remove_diacritics, remove_punctuation, remove_geresh, clean_paren
from homograph_analysis.models import MODEL_PATHS
from homograph_analysis import utils

from homograph_analysis.masked_embedding import MaskedEmbeddingExtractor, extract_and_save_embedding

NA = -1

OVERWRITE = 'o'
ABORT = 'a'
EXISTING = 'e'

def compute_word_index(sample: str, word: str, instance_id: int) -> int:
    """
    Compute the word index (position in sentence.split()) for the k-th occurrence of a word.

    Args:
        sample: The sentence
        word: The word to find
        instance_id: Which occurrence to find (1-indexed, so 1 = first occurrence)

    Returns:
        The index in sample.split() where the word appears

    Raises:
        ValueError: If the word doesn't appear instance_id times in the sample
    """
    words = sample.split()
    occurrence_count = 0

    for i, w in enumerate(words):
        if w == word:
            occurrence_count += 1
            if occurrence_count == instance_id:
                return i

    return NA


def clean_sample_for_word_location(text):
    """ when we search for the homograph, we want to remove distracting features in the text"""
    if not isinstance(text, str):
        raise TypeError(f"Expected string in 'sample' column, got {type(text).__name__}: {text!r}. Check your data for empty/NaN values.")
    return remove_punctuation(text)

def main(
        xlsx_path: str|Path,
        model_path: str,
        sheet_name: str = "samples",
        output_path: Optional[str|Path] = None,
        layer: int = -1,
        clean_punct = True,
        clean_geresh = False,
        instruction = None
):
    """
    Load homograph data and extract masked embeddings for each sample.

    Args:
        model_path: Path to the BERT model directory
        xlsx_path: Path to the XLSX file.
        sheet_name: Name of the sheet/tab to read. Defaults to 'samples'
        output_path: Path for the output JSONL file. Defaults to 'homograph_data/embeddings.jsonl'
        layer: Which hidden layer to extract from. Defaults to -1 (last layer)
        instruction: what to do about existing dataset

    Returns:
        Path to the output JSONL file
    """

    if output_path is None:
        output_path = Path(__file__).parent / "homograph_data" / "embeddings.jsonl"

    output_path = Path(output_path)

    if output_path.exists():
        if instruction is None:
            instruction = input(f"Output file exists. How should I proceed? [(o)verwrite / (e)xisting / (a)bort] (Default: Abort) ")
        if instruction.lower().strip() == OVERWRITE:
            print("Recomputing masked embeddings...")
            output_path.unlink()
        elif instruction.lower().strip() == EXISTING:
            print("Using the existing embeddings file")
            return  # use what we have, no questions asked
        else:
            print("Aborting")
            exit(1)

    print("about to load data...")
    df = load_data(xlsx_path, sheet_name)
    print("data loaded!")

    df['sample'] = df['sample'].apply(clean_paren).apply(remove_diacritics)
    if clean_punct:
        df['sample'] = df['sample'].apply(remove_punctuation)
    if clean_geresh:
        df['sample'] = df['sample'].apply(remove_geresh)

    print(f"Loaded {len(df)} samples")
    df = df[df['ignore'] == False]
    print(f"Removed those marked for 'ignore' - have {len(df)} samples")

    extractor = MaskedEmbeddingExtractor(model_path)

    for idx, row in df.iterrows():
        metadata = {
            "sentence_id": row["id"],
            "meaning_id": row["meaning_id"],
            "word_id": row["word_id"]
        }

        extract_and_save_embedding(
            extractor=extractor,
            sentence=row["sample"],
            word_index=row["word_index"],
            output_path=str(output_path),
            layer=layer,
            metadata=metadata
        )

    print(f"Output saved to: {output_path}")


def load_data(
        xlsx_path: str,
        sheet_name: str = "samples"
) -> pd.DataFrame:
    """
    Load homograph data from an XLSX file and compute word indices.

    Args:
        xlsx_path: Path to the XLSX file. Defaults to 'homograph_data/JA_Homograph_DB.xlsx'
        sheet_name: Name of the sheet/tab to read. Defaults to 'samples'

    Returns:
        DataFrame with columns: sample, word, instance_id, word_index (and any other original columns)
    """

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    df['clean_sample'] = df['sample'].apply(clean_sample_for_word_location)


    df["word_index"] = df.apply(
        lambda row: compute_word_index(row["clean_sample"], row["word"], row["instance_id"]),
        axis=1
    )

    print(f"Found {df[df['word_index'] == NA].shape[0]} cases where there was no matching word found")
    print(df[df['word_index'] == NA])

    df = df[df['word_index'] != NA].copy()

    print(f"Have {df.shape[0]} cases to use for analysis")

    return df

if __name__ == "__main__":

    ################## NOTE ##########################
    ##      USE `pipeline.py` FOR THE FULL FLOW     ##
    ##################################################

    import json
    # config_path = 'jabert_config.json'
    config_path = 'camelbert_ca_config.json'
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    XLSX_PATH = config['dataset_xlsx']
    MODEL_PATH = MODEL_PATHS[config['model'].upper()]
    layer = int(config['layer'])
    output_path = utils.get_embedding_path(config_path)
    main(xlsx_path=XLSX_PATH, model_path=MODEL_PATH, output_path=output_path, layer=layer)
