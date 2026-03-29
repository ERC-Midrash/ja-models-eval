import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from sklearn.metrics import silhouette_score
from homograph_analysis.utils import load_jsonl, filter_dataset
from homograph_analysis.db_utils import save_cluster_run


def compute_silhouette(
        jsonl_path: str,
        word_ids: List[int],
        output_dir: str | Path,
        sample_ids: Optional[List[int]] = None,
        min_samples_per_meaning: int = 2,
        distance: str = 'euclid',
        run_tag: str = '',
        model: Optional[str] = None,
        layer: Optional[int] = None,
        save_to_db: bool = True
) -> List[Dict[str, Any]]:
    """
    Compute silhouette scores on ground-truth labels for all given word_ids.

    This measures how well the GT meaning labels separate in embedding space,
    without doing any clustering. A high silhouette score means the embeddings
    of different meanings are well-separated.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_ids: List of word_ids to process
        output_dir: Directory to save results CSV
        sample_ids: Optionl
        min_samples_per_meaning: Minimum instances required per meaning_id.
            Meanings with fewer samples are excluded.
        distance: Distance metric - 'euclid' (default) or 'cosine'
        run_tag: Tag to append to the output filename
        model: Model name for DB storage (e.g., 'jabert', 'rambert'). If None, DB save is skipped.
        layer: Layer number for DB storage. If None, DB save is skipped.
        save_to_db: Whether to save results to database. Defaults to True.

    Returns:
        List of result dictionaries for each successfully processed word_id
    """
    all_entries = load_jsonl(jsonl_path)
    all_results = []

    metric = 'cosine' if distance == 'cosine' else 'euclidean'

    for word_id in word_ids:
        filtered = filter_dataset(all_entries, word_id, sentence_ids=sample_ids)

        if len(filtered) == 0:
            print(f"Skipping word_id={word_id}: no entries found")
            continue

        word_text = filtered[0]["word"]

        meaning_counts = {}
        for entry in filtered:
            m = entry["meaning_id"]
            meaning_counts[m] = meaning_counts.get(m, 0) + 1

        valid_meanings = [m for m, count in meaning_counts.items() if count >= min_samples_per_meaning]

        excluded_meanings = [m for m in meaning_counts if m not in valid_meanings]
        if excluded_meanings:
            print(f"word_id={word_id} ('{word_text}'): excluding meaning_ids {excluded_meanings} (< {min_samples_per_meaning} samples)")

        k = len(valid_meanings)

        if k <= 1:
            print(f"Skipping word_id={word_id} ('{word_text}'): only {k} meaning(s) after thresholding")
            continue

        filtered_samples = [e for e in filtered if e["meaning_id"] in valid_meanings]

        embeddings = np.array([e["embedding"] for e in filtered_samples])
        meaning_ids = np.array([e["meaning_id"] for e in filtered_samples])

        sil_score = silhouette_score(embeddings, meaning_ids, metric=metric)

        results = {
            "word_id": word_id,
            "word": word_text,
            "k": k,
            "n_samples": len(filtered_samples),
            "distance": distance,
            "silhouette": round(float(sil_score), 4),
        }

        print(f"\n=== Silhouette Score for '{word_text}' (word_id={word_id}) ===")
        print(f"Number of samples: {len(filtered_samples)}")
        print(f"Number of meanings (k): {k}")
        print(f"Distance metric: {distance}")
        print(f"Silhouette score: {sil_score:.4f}")

        all_results.append(results)

    if all_results:
        csv_path = save_results_to_csv(all_results, output_dir, tag=run_tag, distance=distance)

        if save_to_db and model is not None and layer is not None:
            save_silhouette_results_to_db(all_results, model, layer, distance)

    return all_results


def save_results_to_csv(results: List[Dict[str, Any]], output_dir: str | Path, tag: str = '', distance: str = 'euclid') -> Path:
    """
    Save silhouette results to a CSV file.

    Args:
        results: List of result dictionaries from compute_silhouette
        output_dir: Directory where the CSV will be saved
        tag: Tag to append to the filename
        distance: Distance metric used - included in filename

    Returns:
        Path to the saved CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data = []
    for r in results:
        csv_data.append({
            "word": r["word"],
            "word_id": r["word_id"],
            "k": r["k"],
            "n_samples": r["n_samples"],
            "distance": r["distance"],
            "silhouette": r["silhouette"],
        })

    df = pd.DataFrame(csv_data)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"silhouette_results{tag}_{distance}_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8_sig')

    print(f"\nSilhouette results saved to: {csv_path}")
    return csv_path


def save_silhouette_results_to_db(
        results: List[Dict[str, Any]],
        model: str,
        layer: int,
        distance: str,
        db_path: Optional[Path] = None
) -> int:
    """
    Save silhouette results to the database.

    Args:
        results: List of result dictionaries from compute_silhouette
        model: Model name (e.g., 'jabert', 'rambert')
        layer: Layer number
        distance: Distance metric used ('euclid' or 'cosine')
        db_path: Path to the database file. If None, uses the default path.

    Returns:
        The ID of the created run entry
    """
    db_results = []
    for r in results:
        db_results.append({
            'word': r['word'],
            'word_id': r['word_id'],
            'k': r['k'],
            'n_samples': r['n_samples'],
            'gt_clusters': '',
            'pred_clusters': '',
            'metrics': {
                'silhouette': r['silhouette']
            }
        })

    params = {'distance': distance}
    run_id = save_cluster_run(
        clustering_alg='silhouette_gt',
        model=model,
        layer=layer,
        params=params,
        results=db_results,
        db_path=db_path
    )

    print(f"Silhouette results saved to database: run_id={run_id}")
    return run_id


if __name__ == "__main__":
    model = 'jabert'
    layer = -1
    JSONL_PATH = f"homograph_data/embeddings_{model.lower()}_layer{abs(layer)}.jsonl"
    VIS_PATH = "homograph_data/visualizations"
    output_dir = Path(VIS_PATH) / Path(JSONL_PATH).stem

    compute_silhouette(
        jsonl_path=JSONL_PATH,
        word_ids=list(range(1, 17)),
        output_dir=output_dir,
        distance='euclid',
        model=model,
        layer=layer,
        run_tag=f'_{model.lower()}_layer{abs(layer)}',
    )
