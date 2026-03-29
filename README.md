# ja-models-eval

Scripts for evaluating the performance of JA (Judeo-Arabic) BERT models on homograph disambiguation, with supporting tools for Arabic transliteration and error correction.

## Overview

This repo contains two main components:

1. **Homograph Analysis** (`homograph_analysis/`) -- Evaluate how well different BERT models distinguish between meanings of homograph words, using contextual embeddings from masked language models.
2. **Arabic Transliteration & Correction** (`arabic/`) -- Convert Judeo-Arabic text (Hebrew script) to Arabic script, and optionally apply SWEET error correction, to enable evaluation with Arabic-language BERT models.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `transformers`, `torch`, `scikit-learn`, `pandas`, `openpyxl`, `matplotlib`. Optional: `umap-learn` for UMAP visualizations.

BERT models are downloaded from HuggingFace on first use (HeArBERT, mBERT, CAMeLBERT variants). If using a local model (e.g. JABERT), set its path in `homograph_analysis/models.py`.

## Input Data

The pipeline expects an Excel file (`.xlsx`) with homograph word entries -- each row is a sentence containing a target homograph word, annotated with word ID, meaning ID, and sentence ID. The dataset used in this project contains 16 Judeo-Arabic homograph words with 2-3 meanings each.

## Part 1: Homograph Analysis Pipeline

This pipeline extracts contextual embeddings for homograph words from BERT models, then evaluates how well the embeddings separate different meanings using clustering and classification.

### How It Works

1. **Embedding extraction** -- For each sentence, the target word is replaced with `[MASK]` and the model's hidden state at that position is taken as the word's contextual embedding.
2. **Clustering** -- K-means clustering groups embeddings per word (k = number of known meanings). Evaluated with adjusted RAND index and aligned accuracy.
3. **KNN classification** -- Leave-one-out cross-validation with k-nearest neighbors tests whether embeddings of the same meaning are close in the embedding space.
4. **Silhouette scores** -- Measures how well-separated the meaning clusters are (cosine and Euclidean).

### Running the Pipeline

The main entry point is `homograph_analysis/pipeline.py`. It reads a JSON config file that specifies the model, dataset, layer, and output folder.

**Config file format** (see existing `*_config.json` files for examples):

```json
{
  "dataset_xlsx": "homograph_analysis/homograph_data/JA_Homograph_dataset.xlsx",
  "model": "HEARBERT",
  "layer": "-1",
  "outfolder": "homograph_analysis/homograph_data/all_samples"
}
```

- `model` -- Key from `MODEL_PATHS` in `homograph_analysis/models.py`.
- `layer` -- Which transformer layer to extract embeddings from (`"-1"` = last layer).
- `outfolder` -- Where results (CSVs, PNGs, JSONL embeddings) are saved.

**To run:**

```bash
python -m homograph_analysis.pipeline
```

Edit the `__main__` block in `pipeline.py` to point at your config file. The pipeline will:
1. Generate embeddings (saved as `.jsonl`)
2. Run K-means clustering (outputs CSV + visualization PNGs)
3. Run KNN leave-one-out (outputs CSV)
4. Compute silhouette scores (outputs CSV)

Results are also saved to a local SQLite database (`homograph_analysis/homograph.db`) for cross-model comparison.

### Comparing Results Across Models

After running the pipeline for multiple models, use `overall_analysis.py` or `overall_analysis_db.py` to generate comparison box plots and per-word line charts across all configurations.

### Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator -- runs the full evaluation for one model config |
| `models.py` | Model name-to-path mapping |
| `masked_embedding.py` | Embedding extraction (mask target word, read hidden state) |
| `homograph_dataset_processing.py` | Load XLSX, clean text, compute word positions, generate embeddings |
| `clusteration.py` | K-means and agglomerative clustering with evaluation metrics |
| `knn.py` | KNN leave-one-out cross-validation |
| `embedding_topology_metrics.py` | Silhouette score computation |
| `embedding_visualizer.py` | t-SNE and UMAP scatter plots |
| `overall_analysis.py` | Aggregate comparison graphs from CSV files |
| `overall_analysis_db.py` | Aggregate comparison graphs from SQLite DB |
| `db_utils.py` | SQLite schema and read/write helpers |
| `utils.py` | Text cleaning (diacritics, punctuation), JSONL loading, filtering |

## Part 2: Arabic Transliteration & Correction

These scripts convert the Judeo-Arabic homograph dataset from Hebrew script to Arabic script, enabling evaluation with Arabic-language BERT models (e.g. CAMeLBERT). This pipeline is a recreation of the transliteration and SWEET error-correction flows proposed by Moreno Gonzalez et al. (2025) (https://arxiv.org/abs/2507.04746).

### Transliteration

`arabic/transliterate_homograph_dataset.py` reads the Hebrew-script dataset and produces two Arabic-script variants:

- **Dotted** -- Preserves diacritical dots that distinguish letters (e.g. Hebrew gimel with upper dot -> Arabic ghayn).
- **Dotless** -- Strips diacritics for a "bare" Arabic representation.

The character mapping is defined in `arabic/transliterate_ja2arabic.py`.

**To run:**

```bash
python -m arabic.transliterate_homograph_dataset
```

Outputs: `Arabic_Homograph_dotted.xlsx` and `Arabic_Homograph_dotless.xlsx` in the homograph data folder.

### SWEET Error Correction

`arabic/sweet_correct_homograph_dataset.py` applies the SWEET model (CAMeL-Lab's text editing model) to the transliterated Arabic text to correct orthographic errors introduced by transliteration.

The script handles word alignment -- after SWEET rewrites a sentence, it recovers the corrected version of the target homograph word by position or search, tracking the alignment status (`UNCHANGED`, `HOMOG_PRESERVED`, `HOMOG_MOVED`).

**To run:**

```bash
python -m arabic.sweet_correct_homograph_dataset
```

Input: `Arabic_Homograph_dotted.xlsx`. Output: `Arabic_Homograph_sweet_dotted.xlsx`.

The corrected dataset can then be used as input to the homograph analysis pipeline with a CAMeLBERT config.

### Key Files

| File | Purpose |
|------|---------|
| `transliterate_ja2arabic.py` | Hebrew-to-Arabic character mapping (dotted and dotless) |
| `transliterate_homograph_dataset.py` | Batch transliterate the XLSX dataset |
| `sweet_rewrite.py` | SWEET model loading and inference (extracted from CAMeL-Lab) |
| `sweet_correct_homograph_dataset.py` | Apply SWEET to transliterated dataset with word alignment |

## Adapting for Your Own Use

To use this framework with a different dataset or model:

1. **Dataset** -- Prepare an XLSX with columns for sentence text, target word, word ID, meaning ID, and sentence ID matching the expected format.
2. **Model** -- Add your model's HuggingFace ID or local path to `MODEL_PATHS` in `homograph_analysis/models.py`.
3. **Config** -- Create a JSON config pointing to your dataset, model key, desired layer, and output folder.
4. **Run** -- Execute the pipeline as described above.

## Known Issues

- Some paths in `models.py`, `pipeline.py`, and `overall_analysis_db.py` are hardcoded to the original development environment. Update these to match your setup before running.
- Config JSON files use Windows-style backslashes in paths. On Linux/macOS you may need to convert these to forward slashes.
