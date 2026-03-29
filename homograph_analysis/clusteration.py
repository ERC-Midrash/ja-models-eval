import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from homograph_analysis.utils import load_jsonl, filter_dataset
from homograph_analysis.db_utils import save_cluster_run


VERTICAL = 'V'
HORIZONTAL = 'H'


def group_ids_by_label(sample_ids: List, labels: List) -> str:
    """
    Group sample IDs by their label values and return as a string of tuples.

    Args:
        sample_ids: List of sample/sentence IDs
        labels: List of labels (meaning_ids or cluster_labels)

    Returns:
        String representation of grouped IDs, e.g., "[(1, 2, 4), (3, 5, 6)]"
    """
    groups = {}
    for sid, label in zip(sample_ids, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(sid)

    sorted_labels = sorted(groups.keys())
    result = [tuple(sorted(groups[label])) for label in sorted_labels]
    return str(result)


def compute_aligned_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray, k: int) -> tuple:
    """
    Compute accuracy after optimally aligning cluster labels to ground truth.

    K-means cluster IDs are arbitrary, so we find the best mapping from
    cluster IDs to meaning IDs using the Hungarian algorithm.

    Returns:
        Tuple of (accuracy, aligned_predictions)
    """
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)

    cost_matrix = np.zeros((len(unique_pred), len(unique_true)))
    for i, pred in enumerate(unique_pred):
        for j, true in enumerate(unique_true):
            cost_matrix[i, j] = -np.sum((pred_labels == pred) & (true_labels == true))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {unique_pred[row_ind[i]]: unique_true[col_ind[i]] for i in range(len(row_ind))}

    aligned_labels = np.array([mapping.get(p, p) for p in pred_labels])
    accuracy = accuracy_score(true_labels, aligned_labels)

    return accuracy, aligned_labels


def visualize_clusters(
        embeddings: np.ndarray,
        meaning_ids: np.ndarray,
        cluster_labels: np.ndarray,
        word_text: str,
        word_id: int,
        sample_ids: Optional[List] = None,
        orient=HORIZONTAL,
        output_dir: Optional[str] = None
) -> None:
    """
    Create a 2D PCA visualization comparing ground truth vs predicted clusters.

    For small datasets (~10 samples), shows a side-by-side scatter plot with
    annotations for each point.

    Args:
        sample_ids: Optional list of IDs to use for point labels. If None, uses indices.
    """
    if sample_ids is None:
        sample_ids = list(range(len(embeddings)))
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        explained_var = sum(pca.explained_variance_ratio_) * 100
    else:
        coords = embeddings
        explained_var = 100

    if orient == HORIZONTAL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    elif orient == VERTICAL:
        fig, axes = plt.subplots(2, 1, figsize=(7, 12))
    else:
        raise ValueError("orient should be V or H")

    unique_meanings = np.unique(meaning_ids)
    colors_gt = plt.cm.Set1(np.linspace(0, 1, len(unique_meanings)))
    meaning_to_color = {m: colors_gt[i] for i, m in enumerate(unique_meanings)}

    ax1 = axes[0]
    for meaning in unique_meanings:
        mask = meaning_ids == meaning
        ax1.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[meaning_to_color[meaning]],
            label=f"meaning {meaning}",
            s=100, edgecolors='black', linewidths=1
        )

    for i, (x, y) in enumerate(coords):
        ax1.annotate(str(sample_ids[i]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_title(f"Ground Truth (meaning_id)")
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.legend()

    unique_clusters = np.unique(cluster_labels)
    colors_pred = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))
    cluster_to_color = {c: colors_pred[i] for i, c in enumerate(unique_clusters)}

    ax2 = axes[1]
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        ax2.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cluster_to_color[cluster]],
            label=f"cluster {cluster}",
            s=100, edgecolors='black', linewidths=1
        )

    for i, (x, y) in enumerate(coords):
        ax2.annotate(str(sample_ids[i]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_title(f"K-Means Clusters")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.legend()

    fig.suptitle(f"'{word_text}' (word_id={word_id}) - PCA explains {explained_var:.1f}% variance", fontsize=12)
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / f"clusters_word_{word_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def save_results_to_csv(results: List[Dict[str, Any]], output_dir: str | Path, tag:str='', distance: str = 'euclid') -> Path:
    """
    Save clustering results to a CSV file.

    Args:
        results: List of result dictionaries from cluster_all
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
        pred_clustering = group_ids_by_label(r["sample_ids"], r["cluster_labels"])
        csv_data.append({
            "word": r["word"],
            "word_id": r["word_id"],
            "k": r["k"],
            "n_samples": r["n_samples"],
            "n_mistakes": r["n_mistakes"],
            "percent_accurate": round(r["aligned_accuracy"], 3),
            "RAND_index": round(r["adjusted_rand_index"], 3),
            "distance": r.get("distance", distance),
            "gt_clusters": gt_clustering,
            "pred_clusters": pred_clustering
        })

    df = pd.DataFrame(csv_data)
    csv_path = output_dir / f"clustering_results{tag}_{distance}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8_sig')

    print(f"\nClustering results saved to: {csv_path}")
    return csv_path


def cluster_all(
        jsonl_path: str,
        word_ids: List[int],
        output_dir: str|Path,
        min_samples_per_meaning: int = 2,
        run_tag='',
        sample_ids: Optional[List[int]] = None,
        distance: str = 'euclid',
        model: Optional[str] = None,
        layer: Optional[int] = None,
        orient: str = HORIZONTAL,
        save_to_db: bool = True
) -> List[Dict[str, Any]]:
    """
    Run clustering for all given word_ids, determining k from the data.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_ids: List of word_ids to process
        output_dir: Directory to save visualizations and results CSV
        min_samples_per_meaning: Minimum instances required per meaning_id.
            Meanings with fewer samples are excluded from clustering.
        sample_ids: list of sample ids to consider for this run. If None, use all.
        distance: Distance metric to use - 'euclid' (default) or 'cosine'
        model: Model name for DB storage (e.g., 'jabert', 'rambert'). If None, DB save is skipped.
        layer: Layer number for DB storage. If None, DB save is skipped.
        save_to_db: Whether to save results to database. Defaults to True.

    Returns:
        List of result dictionaries for each successfully clustered word_id
    """
    all_entries = load_jsonl(jsonl_path)
    all_results = []

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

        filtered_for_clustering = [e for e in filtered if e["meaning_id"] in valid_meanings]

        embeddings = np.array([e["embedding"] for e in filtered_for_clustering])
        meaning_ids = np.array([e["meaning_id"] for e in filtered_for_clustering])
        # sentences = [e["sentence"] for e in filtered_for_clustering]
        current_sample_ids = [e.get("sentence_id", i) for i, e in enumerate(filtered_for_clustering)]

        embeddings_for_clustering = embeddings

        # For cosine distance with KMeans, normalize embeddings (L2 normalization)
        # This makes Euclidean distance equivalent to cosine distance
        if distance == 'cosine':
            embeddings_for_clustering = normalize(embeddings_for_clustering, norm='l2')

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_for_clustering)

        ari = adjusted_rand_score(meaning_ids, cluster_labels)
        accuracy, aligned_labels = compute_aligned_accuracy(meaning_ids, cluster_labels, k)

        results = {
            "word_id": word_id,
            "word": word_text,
            "n_samples": len(filtered_for_clustering),
            "k": k,
            "distance": distance,
            "adjusted_rand_index": ari,
            "aligned_accuracy": accuracy,
            "n_mistakes": round(len(filtered_for_clustering)*(1-accuracy)),
            "meaning_ids": meaning_ids.tolist(),
            "cluster_labels": cluster_labels.tolist(),
            "aligned_labels": aligned_labels.tolist(),
            "sample_ids": current_sample_ids
        }

        print(f"\n=== Clustering Results for '{word_text}' (word_id={word_id}) ===")
        print(f"Number of samples: {len(filtered_for_clustering)}")
        print(f"Number of clusters (k): {k}")
        print(f"Distance metric: {distance}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Aligned Accuracy: {accuracy:.3f}")

        visualize_clusters(
            embeddings=embeddings,
            meaning_ids=meaning_ids,
            cluster_labels=cluster_labels,
            word_text=word_text,
            word_id=word_id,
            sample_ids=current_sample_ids,
            # sentences=sentences,
            orient=orient,
            output_dir=output_dir
        )

        all_results.append(results)

    if all_results:
        from datetime import datetime
        save_results_to_csv(all_results, output_dir, tag=run_tag + '_kmeans_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), distance=distance)

        if save_to_db and model is not None and layer is not None:
            save_clustering_results_to_db(
                results=all_results,
                clustering_alg='kmeans',
                model=model,
                layer=layer,
                distance=distance,
            )

    return all_results


def visualize_dendrogram(
        embeddings: np.ndarray,
        meaning_ids: np.ndarray,
        word_text: str,
        word_id: int,
        sample_ids: List,
        linkage_method: str = "ward",
        output_dir: Optional[str] = None
) -> None:
    """
    Create a dendrogram visualization with color-coded labels by meaning_id.

    Args:
        embeddings: Array of embeddings to cluster
        meaning_ids: Ground truth meaning IDs for color-coding
        word_text: The word being analyzed
        word_id: The word_id for the filename
        sample_ids: List of sample IDs for leaf labels
        linkage_method: Linkage method for hierarchical clustering
        output_dir: Directory to save visualization. If None, displays instead.
    """
    Z = scipy_linkage(embeddings, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 6))

    unique_meanings = np.unique(meaning_ids)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_meanings)))
    meaning_to_color = {m: colors[i] for i, m in enumerate(unique_meanings)}

    label_colors = {i: meaning_to_color[meaning_ids[i]] for i in range(len(meaning_ids))}

    labels = [str(sid) for sid in sample_ids]

    dendro = dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10
    )

    leaf_order = dendro['leaves']
    x_labels = ax.get_xticklabels()
    for i, label in enumerate(x_labels):
        original_idx = leaf_order[i]
        label.set_color(label_colors[original_idx])
        label.set_fontweight('bold')

    legend_handles = []
    for meaning in unique_meanings:
        handle = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=meaning_to_color[meaning],
                           markersize=10, label=f'meaning {meaning}')
        legend_handles.append(handle)
    ax.legend(handles=legend_handles, loc='upper right')

    ax.set_title(f"Dendrogram: '{word_text}' (word_id={word_id}) - {linkage_method} linkage")
    ax.set_xlabel("Sample ID (colored by meaning_id)")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / f"dendrogram_word_{word_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Dendrogram saved to: {output_path}")
    else:
        plt.show()

    plt.close()

def agglomerative_cluster_all(
        jsonl_path: str,
        word_ids: List[int],
        output_dir: str | Path,
        sample_ids: Optional[List[int]] = None,
        min_samples_per_meaning: int = 2,
        linkage: str = "ward",
        run_tag='',
        distance: str = 'euclid',
        model: Optional[str] = None,
        layer: Optional[int] = None,
        save_to_db: bool = True
) -> List[Dict[str, Any]]:
    """
    Run agglomerative clustering for all given word_ids, determining k from the data.

    Args:
        jsonl_path: Path to the JSONL file with embeddings
        word_ids: List of word_ids to process
        output_dir: Directory to save visualizations and results CSV
        min_samples_per_meaning: Minimum instances required per meaning_id.
            Meanings with fewer samples are excluded from clustering.
        linkage: Linkage method for hierarchical clustering (default: ward)
        distance: Distance metric to use - 'euclid' (default) or 'cosine'
        model: Model name for DB storage (e.g., 'jabert', 'rambert'). If None, DB save is skipped.
        layer: Layer number for DB storage. If None, DB save is skipped.
        save_to_db: Whether to save results to database. Defaults to True.

    Returns:
        List of result dictionaries for each successfully clustered word_id
    """
    all_entries = load_jsonl(jsonl_path)
    all_results = []

    # Ward linkage only works with Euclidean distance
    effective_linkage = linkage
    if distance == 'cosine' and linkage == 'ward':
        print("Warning: ward linkage does not support cosine distance, using 'average' linkage instead")
        effective_linkage = 'average'

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

        filtered_for_clustering = [e for e in filtered if e["meaning_id"] in valid_meanings]

        embeddings = np.array([e["embedding"] for e in filtered_for_clustering])
        meaning_ids = np.array([e["meaning_id"] for e in filtered_for_clustering])
        current_sample_ids = [e.get("sentence_id", i) for i, e in enumerate(filtered_for_clustering)]

        embeddings_for_clustering = embeddings

        if distance == 'cosine':
            agg = AgglomerativeClustering(n_clusters=k, linkage=effective_linkage, metric='cosine')
        else:
            agg = AgglomerativeClustering(n_clusters=k, linkage=effective_linkage)
        cluster_labels = agg.fit_predict(embeddings_for_clustering)

        ari = adjusted_rand_score(meaning_ids, cluster_labels)
        accuracy, aligned_labels = compute_aligned_accuracy(meaning_ids, cluster_labels, k)

        results = {
            "word_id": word_id,
            "word": word_text,
            "n_samples": len(filtered_for_clustering),
            "k": k,
            "linkage": effective_linkage,
            "distance": distance,
            "adjusted_rand_index": ari,
            "aligned_accuracy": accuracy,
            "n_mistakes": round(len(filtered_for_clustering) * (1 - accuracy)),
            "meaning_ids": meaning_ids.tolist(),
            "cluster_labels": cluster_labels.tolist(),
            "aligned_labels": aligned_labels.tolist(),
            "sample_ids": current_sample_ids
        }

        print(f"\n=== Agglomerative Clustering Results for '{word_text}' (word_id={word_id}) ===")
        print(f"Number of samples: {len(filtered_for_clustering)}")
        print(f"Number of clusters (k): {k}")
        print(f"Linkage method: {effective_linkage}")
        print(f"Distance metric: {distance}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Aligned Accuracy: {accuracy:.3f}")

        visualize_dendrogram(
            embeddings=embeddings,
            meaning_ids=meaning_ids,
            word_text=word_text,
            word_id=word_id,
            sample_ids=current_sample_ids,
            linkage_method=effective_linkage,
            output_dir=output_dir
        )

        all_results.append(results)

    if all_results:
        from datetime import datetime
        save_results_to_csv(
            all_results,
            output_dir,
            tag=f"_{run_tag}_agglomerative_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            distance=distance
        )

        if save_to_db and model is not None and layer is not None:
            save_clustering_results_to_db(
                results=all_results,
                clustering_alg='agglomerative',
                model=model,
                layer=layer,
                distance=distance,
                linkage=effective_linkage,
            )

    return all_results


def save_clustering_results_to_db(
        results: List[Dict[str, Any]],
        clustering_alg: str,
        model: str,
        layer: int,
        distance: str,
        linkage: Optional[str] = None,
        db_path: Optional[Path] = None
) -> int:
    """
    Save clustering results to the database.

    Args:
        results: List of result dictionaries from cluster_all or agglomerative_cluster_all
        clustering_alg: Algorithm name ('kmeans' or 'agglomerative')
        model: Model name (e.g., 'jabert', 'rambert')
        layer: Layer number
        distance: Distance metric used ('euclid' or 'cosine')
        linkage: Linkage method (for agglomerative clustering)
        db_path: Path to the database file. If None, uses the default path.

    Returns:
        The ID of the created run entry
    """
    db_results = []
    for r in results:
        gt_clustering = group_ids_by_label(r["sample_ids"], r["meaning_ids"])
        pred_clustering = group_ids_by_label(r["sample_ids"], r["cluster_labels"])
        db_results.append({
            'word': r['word'],
            'word_id': r['word_id'],
            'k': r['k'],
            'n_samples': r['n_samples'],
            'gt_clusters': gt_clustering,
            'pred_clusters': pred_clustering,
            'metrics': {
                'n_mistakes': r['n_mistakes'],
                'percent_accurate': round(r['aligned_accuracy'], 3),
                'RAND_index': round(r['adjusted_rand_index'], 3)
            }
        })

    params = {'distance': distance}
    if linkage is not None:
        params['linkage'] = linkage
    run_id = save_cluster_run(
        clustering_alg=clustering_alg,
        model=model,
        layer=layer,
        params=params,
        results=db_results,
        db_path=db_path
    )

    print(f"Clustering results saved to database: run_id={run_id}")
    return run_id


if __name__ == "__main__":
    model = 'jabert'
    layer = -1
    JSONL_PATH = f"homograph_data/embeddings_{model.lower()}_layer{abs(layer)}.jsonl"

    VIS_PATH = f"homograph_data/visualizations"
    output_dir = Path(VIS_PATH) / Path(JSONL_PATH).stem
    # WORD_ID = 3
    #
    # results = cluster_and_evaluate(
    #     jsonl_path=JSONL_PATH,
    #     word_id=WORD_ID,
    #     k=2,
    #     output_dir="homograph_data/visualizations"
    # )
    #
    cluster_all(
        jsonl_path=JSONL_PATH,
        word_ids=list(range(1, 17)),
        output_dir=output_dir,
        distance='cosine',
        model=model,
        layer=layer,
        run_tag=f'_{model.lower()}_layer{abs(layer)}',
        orient=VERTICAL
    )


    # cluster_all(JSONL_PATH, word_ids, output_dir=vis_output_dir, run_tag=f'_{model.lower()}_layer{abs(layer)}',
    #             model=model, layer=layer, distance='cosine')

    # cluster_all(JSONL_PATH, list(range(1, 17)), output_dir=output_dir, distance='cosine', model=model, layer=layer)