import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualizations will be skipped.")


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all entries from a JSONL file."""
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def filter_by_word_id(entries: List[Dict[str, Any]], word_id: int) -> List[Dict[str, Any]]:
    """Filter entries to only those with the given word_id."""
    return [e for e in entries if e.get("word_id") == word_id]


def visualize_embeddings_tsne(
        embeddings: np.ndarray,
        meaning_ids: np.ndarray,
        word_text: str,
        word_id: int,
        sample_ids: List,
        perplexity: float = None,
        output_dir: Optional[str | Path] = None
) -> None:
    """
    Create a t-SNE visualization of embeddings, color-coded by meaning_id.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        meaning_ids: Ground truth meaning IDs for color-coding
        word_text: The word being analyzed
        word_id: The word_id for the filename
        sample_ids: List of sample IDs for point labels
        perplexity: t-SNE perplexity. If None, automatically set based on sample size.
        output_dir: Directory to save visualization. If None, displays instead.
    """
    n_samples = len(embeddings)

    if perplexity is None:
        perplexity = min(30, max(5, n_samples - 1))

    if n_samples < 2:
        print(f"Skipping t-SNE for word_id={word_id}: need at least 2 samples")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_meanings = np.unique(meaning_ids)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_meanings), 2)))
    meaning_to_color = {m: colors[i] for i, m in enumerate(unique_meanings)}

    for meaning in unique_meanings:
        mask = meaning_ids == meaning
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[meaning_to_color[meaning]],
            label=f"meaning {meaning}",
            s=120, edgecolors='black', linewidths=1, alpha=0.8
        )

    for i, (x, y) in enumerate(coords):
        ax.annotate(
            str(sample_ids[i]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )

    ax.set_title(f"t-SNE: '{word_text}' (word_id={word_id})\nperplexity={perplexity:.1f}", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / f"tsne_word_{word_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_embeddings_umap(
        embeddings: np.ndarray,
        meaning_ids: np.ndarray,
        word_text: str,
        word_id: int,
        sample_ids: List,
        n_neighbors: int = None,
        min_dist: float = 0.1,
        output_dir: Optional[str | Path] = None
) -> None:
    """
    Create a UMAP visualization of embeddings, color-coded by meaning_id.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        meaning_ids: Ground truth meaning IDs for color-coding
        word_text: The word being analyzed
        word_id: The word_id for the filename
        sample_ids: List of sample IDs for point labels
        n_neighbors: UMAP n_neighbors parameter. If None, automatically set based on sample size.
        min_dist: UMAP min_dist parameter
        output_dir: Directory to save visualization. If None, displays instead.
    """
    if not UMAP_AVAILABLE:
        print(f"Skipping UMAP for word_id={word_id}: umap-learn not installed")
        return

    n_samples = len(embeddings)

    if n_neighbors is None:
        n_neighbors = min(15, max(2, n_samples - 1))

    if n_samples < 2:
        print(f"Skipping UMAP for word_id={word_id}: need at least 2 samples")
        return

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_meanings = np.unique(meaning_ids)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_meanings), 2)))
    meaning_to_color = {m: colors[i] for i, m in enumerate(unique_meanings)}

    for meaning in unique_meanings:
        mask = meaning_ids == meaning
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[meaning_to_color[meaning]],
            label=f"meaning {meaning}",
            s=120, edgecolors='black', linewidths=1, alpha=0.8
        )

    for i, (x, y) in enumerate(coords):
        ax.annotate(
            str(sample_ids[i]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )

    ax.set_title(f"UMAP: '{word_text}' (word_id={word_id})\nn_neighbors={n_neighbors}, min_dist={min_dist}", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / f"umap_word_{word_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"UMAP visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_word(
        jsonl_path: str,
        word_id: int,
        output_dir: Optional[str | Path] = None,
        methods: List[str] = None,
        min_samples_per_meaning: int = 2
) -> None:
    """
    Create t-SNE and/or UMAP visualizations for a single word_id.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_id: The word_id to visualize
        output_dir: Directory to save visualizations. If None, displays instead.
        methods: List of methods to use ('tsne', 'umap'). If None, uses both.
        min_samples_per_meaning: Minimum samples required per meaning_id
    """
    if methods is None:
        methods = ['tsne', 'umap']

    entries = load_jsonl(jsonl_path)
    filtered = filter_by_word_id(entries, word_id)

    if len(filtered) == 0:
        print(f"Skipping word_id={word_id}: no entries found")
        return

    word_text = filtered[0]["word"]

    meaning_counts = {}
    for entry in filtered:
        m = entry["meaning_id"]
        meaning_counts[m] = meaning_counts.get(m, 0) + 1

    valid_meanings = [m for m, count in meaning_counts.items() if count >= min_samples_per_meaning]

    if len(valid_meanings) <= 1:
        print(f"Skipping word_id={word_id} ('{word_text}'): only {len(valid_meanings)} meaning(s) after thresholding")
        return

    filtered_entries = [e for e in filtered if e["meaning_id"] in valid_meanings]

    embeddings = np.array([e["embedding"] for e in filtered_entries])
    meaning_ids = np.array([e["meaning_id"] for e in filtered_entries])
    sample_ids = [e.get("sentence_id", i) for i, e in enumerate(filtered_entries)]

    print(f"\n=== Visualizing '{word_text}' (word_id={word_id}) ===")
    print(f"Number of samples: {len(filtered_entries)}")
    print(f"Number of meanings: {len(valid_meanings)}")

    if 'tsne' in methods:
        visualize_embeddings_tsne(
            embeddings=embeddings,
            meaning_ids=meaning_ids,
            word_text=word_text,
            word_id=word_id,
            sample_ids=sample_ids,
            output_dir=output_dir
        )

    if 'umap' in methods:
        visualize_embeddings_umap(
            embeddings=embeddings,
            meaning_ids=meaning_ids,
            word_text=word_text,
            word_id=word_id,
            sample_ids=sample_ids,
            output_dir=output_dir
        )


def visualize_all(
        jsonl_path: str,
        word_ids: List[int],
        output_dir: str | Path,
        methods: List[str] = None,
        min_samples_per_meaning: int = 2
) -> None:
    """
    Create t-SNE and/or UMAP visualizations for all given word_ids.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_ids: List of word_ids to visualize
        output_dir: Directory to save visualizations
        methods: List of methods to use ('tsne', 'umap'). If None, uses both.
        min_samples_per_meaning: Minimum samples required per meaning_id
    """
    for word_id in word_ids:
        visualize_word(
            jsonl_path=jsonl_path,
            word_id=word_id,
            output_dir=output_dir,
            methods=methods,
            min_samples_per_meaning=min_samples_per_meaning
        )


if __name__ == "__main__":
    JSONL_PATH = "homograph_data/embeddings_jabert_layer1.jsonl"
    VIS_PATH = "homograph_data/visualizations"
    output_dir = Path(VIS_PATH) / Path(JSONL_PATH).stem

    visualize_all(
        JSONL_PATH,
        list(range(1, 17)),
        output_dir=output_dir,
        methods=['tsne', 'umap']
    )
