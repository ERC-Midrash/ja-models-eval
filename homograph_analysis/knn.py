import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from datetime import datetime
from collections import defaultdict
from homograph_analysis.utils import load_jsonl, filter_dataset
from homograph_analysis.db_utils import save_cluster_run


def knn_loocv_all(
        jsonl_path: str,
        word_ids: List[int],
        output_dir: str | Path,
        sample_ids: Optional[List[int]] = None,
        n_neighbors: int = 1,
        min_samples_per_meaning: int = 2,
        run_tag: str = '',
        distance: str = 'euclid',
        model: Optional[str] = None,
        layer: Optional[int] = None,
        save_to_db: bool = True
) -> List[Dict[str, Any]]:
    """
    Run KNN LOOCV classification for all given word_ids.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_ids: List of word_ids to process
        output_dir: Directory to save results CSV
        sample_ids: Optional. List of sample ids to use for this analysis
        n_neighbors: Number of neighbors for KNN (default: 1)
        min_samples_per_meaning: Minimum instances required per meaning_id
        run_tag: Tag to append to the output filename
        distance: Distance metric to use - 'euclid' (default) or 'cosine'
        model: Model name for DB storage (e.g., 'jabert', 'rambert'). If None, DB save is skipped.
        layer: Layer number for DB storage. If None, DB save is skipped.
        save_to_db: Whether to save results to database. Defaults to True.

    Returns:
        List of result dictionaries for each successfully classified word_id
    """
    all_results = []

    for word_id in word_ids:
        result = knn_loocv_evaluate(
            jsonl_path=jsonl_path,
            word_id=word_id,
            sample_ids=sample_ids,
            n_neighbors=n_neighbors,
            min_samples_per_meaning=min_samples_per_meaning,
            distance=distance
        )

        if result is not None:
            all_results.append(result)

    if all_results:
        save_results_to_csv(all_results, output_dir, tag=run_tag, distance=distance)

        if save_to_db and model is not None and layer is not None:
            save_knn_results_to_db(all_results, model, layer, distance, n_neighbors)

    return all_results


def knn_loocv_evaluate(
        jsonl_path: str,
        word_id: int,
        sample_ids: Optional[List[int]] = None,
        n_neighbors: int = 1,
        min_samples_per_meaning: int = 2,
        distance: str = 'euclid'
) -> Optional[Dict[str, Any]]:
    """
    Run KNN with Leave-One-Out Cross Validation for a given word_id.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_id: The word_id to filter and classify
        sample_ids: optional. List of ids to filter by
        n_neighbors: Number of neighbors for KNN (default: 1)
        min_samples_per_meaning: Minimum samples required per meaning_id
        distance: Distance metric to use - 'euclid' (default) or 'cosine'

    Returns:
        Dictionary with evaluation metrics, or None if skipped
    """
    entries = load_jsonl(jsonl_path)
    filtered = filter_dataset(entries, word_id, sentence_ids=sample_ids)

    if len(filtered) == 0:
        print(f"Skipping word_id={word_id}: no entries found")
        return None

    word_text = filtered[0]["word"]

    meaning_counts = defaultdict(int)
    for entry in filtered:
        m = entry["meaning_id"]
        meaning_counts[m] += 1

    # only work with meanings with some minimal representation in the dataset
    valid_meanings = [m for m, count in meaning_counts.items() if count >= min_samples_per_meaning]

    excluded_meanings = [m for m in meaning_counts if m not in valid_meanings]
    if excluded_meanings:
        print(f"word_id={word_id} ('{word_text}'): excluding meaning_ids {excluded_meanings} (< {min_samples_per_meaning} samples)")

    k = len(valid_meanings)

    if k <= 1:
        print(f"Skipping word_id={word_id} ('{word_text}'): only {k} meaning(s) after thresholding")
        return None

    filtered_for_classification = [e for e in filtered if e["meaning_id"] in valid_meanings]

    embeddings = np.array([e["embedding"] for e in filtered_for_classification])
    meaning_ids = np.array([e["meaning_id"] for e in filtered_for_classification])
    sample_ids = [e["sentence_id"] for e in filtered_for_classification]

    effective_k = min(n_neighbors, len(embeddings) - 1)  # todo: not sure I agree with this logic - should n_neighbours not be a strict threshold?
    if effective_k < 1:
        print(f"Skipping word_id={word_id} ('{word_text}'): not enough samples for LOOCV")
        return None

    loo = LeaveOneOut()
    predictions = []

    metric = 'cosine' if distance == 'cosine' else 'euclidean'

    for train_idx, test_idx in loo.split(embeddings):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train = meaning_ids[train_idx]

        knn = KNeighborsClassifier(n_neighbors=effective_k, metric=metric)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        predictions.append(pred[0])

    predictions = np.array(predictions)
    accuracy = accuracy_score(meaning_ids, predictions)
    n_mistakes = int(np.sum(predictions != meaning_ids))

    results = {
        "word_id": word_id,
        "word": word_text,
        "n_samples": len(filtered_for_classification),
        "k_neighbors": effective_k,
        "n_meanings": k,
        "accuracy": accuracy,
        "n_mistakes": n_mistakes,
        "distance": distance,
        "meaning_ids": meaning_ids.tolist(),
        "predictions": predictions.tolist(),
        "sample_ids": sample_ids
    }

    print(f"\n=== KNN LOOCV Results for '{word_text}' (word_id={word_id}) ===")
    print(f"Number of samples: {len(filtered_for_classification)}")
    print(f"Number of meanings: {k}")
    print(f"K neighbors: {effective_k}")
    print(f"Distance metric: {distance}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Mistakes: {n_mistakes}")

    return results


def group_ids_by_label(sample_ids: List, labels: List) -> str:
    """
    Group sample IDs by their label values and return as a string of tuples.

    Args:
        sample_ids: List of sample/sentence IDs
        labels: List of labels (meaning_ids or predicted_labels)

    Returns:
        String representation of grouped IDs, e.g., "[(1, 2, 4), (3, 5, 6)]"
    """
    groups = defaultdict(list)
    for sid, label in zip(sample_ids, labels):
        groups[label].append(sid)

    sorted_labels = sorted(groups.keys())
    result = [tuple(sorted(groups[label])) for label in sorted_labels]
    return str(result)


def save_results_to_csv(results: List[Dict[str, Any]], output_dir: str | Path, tag: str = '', distance: str = 'euclid') -> Path:
    """
    Save KNN classification results to a CSV file.

    Args:
        results: List of result dictionaries from knn_loocv_all
        output_dir: Directory where the CSV will be saved
        tag: Tag to append to the filename
        distance: Distance metric used - included in filename and as column

    Returns:
        Path to the saved CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data = []
    for r in results:
        gt_clustering = group_ids_by_label(r["sample_ids"], r["meaning_ids"])
        pred_clustering = group_ids_by_label(r["sample_ids"], r["predictions"])
        csv_data.append({
            "word": r["word"],
            "word_id": r["word_id"],
            "k_neighbors": r["k_neighbors"],
            "n_meanings": r["n_meanings"],
            "n_samples": r["n_samples"],
            "n_mistakes": r["n_mistakes"],
            "percent_accurate": round(r["accuracy"], 3),
            "distance": r.get("distance", distance),
            "gt_clusters": gt_clustering,
            "pred_clusters": pred_clustering
        })

    df = pd.DataFrame(csv_data)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = output_dir / f"knn_results{tag}_{distance}_knn_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8_sig')

    print(f"\nKNN results saved to: {csv_path}")
    return csv_path


def save_knn_results_to_db(
        results: List[Dict[str, Any]],
        model: str,
        layer: int,
        distance: str,
        n_neighbors: int,
        db_path: Optional[Path] = None
) -> int:
    """
    Save KNN classification results to the database.

    Args:
        results: List of result dictionaries from knn_loocv_all
        model: Model name (e.g., 'jabert', 'rambert')
        layer: Layer number
        distance: Distance metric used ('euclid' or 'cosine')
        n_neighbors: Number of neighbors used in KNN
        db_path: Path to the database file. If None, uses the default path.

    Returns:
        The ID of the created run entry
    """
    db_results = []
    for r in results:
        gt_clustering = group_ids_by_label(r["sample_ids"], r["meaning_ids"])
        pred_clustering = group_ids_by_label(r["sample_ids"], r["predictions"])
        db_results.append({
            'word': r['word'],
            'word_id': r['word_id'],
            'k': r['k_neighbors'],
            'n_samples': r['n_samples'],
            'gt_clusters': gt_clustering,
            'pred_clusters': pred_clustering,
            'metrics': {
                'n_mistakes': r['n_mistakes'],
                'percent_accurate': round(r['accuracy'], 3)
            }
        })

    params = {
        'distance': distance,
        'k_neighbors': n_neighbors
    }
    run_id = save_cluster_run(
        clustering_alg='knn',
        model=model,
        layer=layer,
        params=params,
        results=db_results,
        db_path=db_path
    )

    print(f"KNN results saved to database: run_id={run_id}")
    return run_id


if __name__ == "__main__":

    model = 'mbert'
    layer = -1
    JSONL_PATH = f"homograph_data/embeddings_{model}_layer{abs(layer)}.jsonl"
    VIS_PATH = "homograph_data/visualizations"
    output_dir = Path(VIS_PATH) / Path(JSONL_PATH).stem

    knn_loocv_all(
        JSONL_PATH,
        list(range(1, 17)),
        output_dir=output_dir,
        n_neighbors=3,
        run_tag='_jabert_layer1_n3',
        distance='cosine',
        model=model,
        layer=layer
    )
