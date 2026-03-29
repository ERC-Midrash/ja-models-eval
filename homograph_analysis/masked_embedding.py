import json
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel


class MaskedEmbeddingExtractor:
    """
    Extracts contextualized embeddings of masked words from BERT models.

    Designed for JABert/RamBert/Gereshless models but works with any BERT-like model.
    """

    def __init__(self, model_path: str):
        """
        Initialize the extractor with a BERT model.

        Args:
            model_path: Path to the BERT model directory or HuggingFace model ID
        """
        self.model_path = model_path

        # For local paths, use directory name; for HuggingFace IDs, keep full ID
        path_obj = Path(model_path)
        if path_obj.is_dir():
            self.model_name = path_obj.name
        else:
            self.model_name = model_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()


    def get_masked_word_embedding(
            self,
            sentence: str,
            word_index: int,
            layer: int = -1
    ) -> torch.Tensor:
        """
        Get the contextualized embedding of a masked word in a sentence.

        The word at the given index is replaced with a single [MASK] token,
        regardless of how many wordpieces it would normally tokenize to.

        Args:
            sentence: The input sentence
            word_index: Index of the word to mask (0-indexed, based on sentence.split())
            layer: Which hidden layer to extract (-1 for last, -2 for second-to-last, etc.)

        Returns:
            Embedding tensor of shape (hidden_size,)

        Raises:
            IndexError: If word_index is out of bounds
        """
        words = sentence.split()

        if word_index < 0 or word_index >= len(words):
            raise IndexError(f"word_index {word_index} out of bounds for sentence with {len(words)} words")

        masked_words = words.copy()
        masked_words[word_index] = self.tokenizer.mask_token
        masked_sentence = " ".join(masked_words)

        inputs = self.tokenizer(masked_sentence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_positions) == 0:
            raise ValueError("Could not find [MASK] token in tokenized sentence")

        mask_pos = mask_positions[0].item()

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        embedding = hidden_states[layer][0, mask_pos, :]

        return embedding


    def get_word_at_index(self, sentence: str, word_index: int) -> str:
        """Get the word at the given index in the sentence."""
        words = sentence.split()
        if word_index < 0 or word_index >= len(words):
            raise IndexError(f"word_index {word_index} out of bounds for sentence with {len(words)} words")
        return words[word_index]


def extract_and_save_embedding(
        extractor: MaskedEmbeddingExtractor,
        sentence: str,
        word_index: int,
        output_path: str,
        layer: int = -1,
        metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Extract a masked word embedding and append it to a JSONL file.

    Args:
        extractor: MaskedEmbeddingExtractor instance
        sentence: The input sentence
        word_index: Index of the word to mask
        output_path: Path to the JSONL output file (will append)
        layer: Which hidden layer to extract from
        metadata: Optional dict of additional key-values to include in the JSONL entry
    """
    word = extractor.get_word_at_index(sentence, word_index)
    embedding = extractor.get_masked_word_embedding(sentence, word_index, layer)

    record = {
        "sentence": sentence,
        "word_index": word_index,
        "word": word,
        "model_name": extractor.model_name,
        "embedding": embedding.cpu().tolist(),
        "layer": layer
    }

    if metadata:
        record.update(metadata)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    MODEL_PATH = os.environ.get('JABERT_MODEL_PATH', 'path/to/local/JABERT')  # Set JABERT_MODEL_PATH env var or edit this

    sentence = "הכתאב דא מליח כתיר"
    word_index = 2

    extractor = MaskedEmbeddingExtractor(MODEL_PATH)

    embedding = extractor.get_masked_word_embedding(
        sentence=sentence,
        word_index=word_index,
        layer=-1
    )

    print(f"Sentence: {sentence}")
    print(f"Masked word: {extractor.get_word_at_index(sentence, word_index)}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 values): {embedding[:10]}")

    extract_and_save_embedding(
        extractor=extractor,
        sentence=sentence,
        word_index=word_index,
        output_path="./test_embeddings.jsonl",
        layer=-1
    )
    print(f"\nSaved embedding to test_embeddings.jsonl")
