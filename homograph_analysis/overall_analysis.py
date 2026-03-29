import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import re


def find_clustering_csvs(base_dir: str | Path) -> List[Path]:
    """
    Find all clustering result CSV files in the visualizations directory.

    Args:
        base_dir: Base directory containing the visualizations folder

    Returns:
        List of paths to CSV files
    """
    base_dir = Path(base_dir)
    csv_files = list(base_dir.glob("**/clustering_results*.csv"))
    return sorted(csv_files)


def parse_csv_metadata(csv_path: Path) -> Dict[str, str]:
    """
    Parse model name, layer, algorithm, and timestamp from CSV filename.

    Expected patterns:
    - clustering_results_<model>_layer<int>_kmeans_<timestamp>.csv
    - clustering_results__<model>_layer<int>_agglomerative_<timestamp>.csv

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with 'model', 'layer', 'algorithm', 'timestamp' keys
    """
    filename = csv_path.stem

    if "agglomerative" in filename:
        algorithm = "agglomerative"
    elif "kmeans" in filename:
        algorithm = "kmeans"
    else:
        algorithm = "unknown"

    layer_match = re.search(r"layer(\d+)", filename)
    layer = layer_match.group(1) if layer_match else "unknown"

    for model in ["jabert", "rambert", "gereshless", "mbert", "hearbert"]:
        if model in filename.lower():
            break
    else:
        model = "unknown"

    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", filename)
    timestamp = timestamp_match.group(1) if timestamp_match else "0000-00-00_00-00-00"

    return {
        "model": model,
        "layer": layer,
        "algorithm": algorithm,
        "timestamp": timestamp
    }


def load_all_results(base_dir: str | Path) -> pd.DataFrame:
    """
    Load all clustering CSVs and combine into a single DataFrame with metadata.

    For each (model, layer, algorithm) combination, only loads the most recent CSV
    based on the timestamp in the filename.

    Args:
        base_dir: Base directory containing the visualizations folder

    Returns:
        Combined DataFrame with model, layer, algorithm columns added
    """
    csv_files = find_clustering_csvs(base_dir)

    csv_with_metadata = []
    for csv_path in csv_files:
        metadata = parse_csv_metadata(csv_path)
        csv_with_metadata.append((csv_path, metadata))

    latest_csvs = {}
    for csv_path, metadata in csv_with_metadata:
        key = (metadata["model"], metadata["layer"], metadata["algorithm"])
        if key not in latest_csvs or metadata["timestamp"] > latest_csvs[key][1]["timestamp"]:
            latest_csvs[key] = (csv_path, metadata)

    all_dfs = []
    for csv_path, metadata in latest_csvs.values():
        df = pd.read_csv(csv_path)
        df["model"] = metadata["model"]
        df["layer"] = metadata["layer"]
        df["algorithm"] = metadata["algorithm"]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def create_comparison_graphs(df: pd.DataFrame, output_path: str | Path = None) -> None:
    """
    Create 3 vertically stacked graphs comparing metrics across models/layers/algorithms.

    X-axis: (model, layer, algorithm) tuples
    Y-axis: n_mistakes, percent_accurate, RAND_index (one per graph)

    Args:
        df: Combined DataFrame with all clustering results
        output_path: Path to save the figure. If None, displays instead.
    """
    df["config"] = df["model"] + "\n" + "L" + df["layer"] + "\n" + df["algorithm"]

    config_order = []
    for model in ["jabert", "mbert", "hearbert"]:  # ["jabert", "rambert", "gereshless"]
        for layer in ["1"]:  # removed layer 2
            for alg in ["kmeans", "agglomerative"]:
                config_order.append(f"{model}\nL{layer}\n{alg}")

    # metrics = ["n_mistakes", "percent_accurate", "RAND_index"]
    metrics = ["percent_accurate", "RAND_index"]
    # metric_labels = ["Number of Mistakes", "Accuracy (%)", "Adjusted RAND Index"]
    metric_labels = ["Accuracy (%)", "Adjusted RAND Index"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    palette = sns.color_palette("husl", n_colors=len(config_order))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        sns.boxplot(
            data=df,
            x="config",
            y=metric,
            hue="config",
            order=config_order,
            hue_order=config_order,
            palette=palette,
            legend=False,
            ax=ax
        )

        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(axis='x', rotation=0, labelsize=8)

        ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)

    axes[0].set_title("Homograph Discrimination Performance Comparison", fontsize=14, fontweight='bold')

    axes[-1].set_xlabel("Configuration (Model / Layer / Algorithm)", fontsize=11)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def create_per_word_comparison_graphs(
        df: pd.DataFrame,
        word_ids: List[int] = None,
        output_path: str | Path = None
) -> None:
    """
    Create plots showing metric values for each word_id across model & layer configurations.

    For each clustering method, creates a row of plots (one per metric).
    Each plot shows lines for each word_id, with x-axis being (model, layer).

    Args:
        df: Combined DataFrame with all clustering results (must have model, layer, algorithm columns)
        word_ids: List of word_ids to include. If None, uses all word_ids in the data.
        output_path: Path to save the figure. If None, displays instead.
    """
    df = df.copy()
    df["config"] = df["model"] + "_L" + df["layer"]

    config_order = []
    for model in ["jabert", "hearbert", "mbert"]:  ## ["jabert", "rambert", "gereshless"]
        for layer in ["1"]: # ["1", "2"]
            config_order.append(f"{model}_L{layer}")

    df = df[df["config"].isin(config_order)]

    if word_ids is None:
        word_ids = sorted(df["word_id"].unique().tolist())
    else:
        df = df[df["word_id"].isin(word_ids)]

    clustering_methods = sorted(df["algorithm"].unique().tolist())
    metrics = ["n_mistakes", "percent_accurate", "RAND_index"]
    metric_labels = ["Number of Mistakes", "Accuracy", "RAND Index"]

    n_rows = len(clustering_methods)
    n_cols = len(metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    palette = sns.color_palette("husl", n_colors=len(word_ids))
    word_id_to_color = {wid: palette[i] for i, wid in enumerate(word_ids)}

    for row_idx, alg in enumerate(clustering_methods):
        alg_df = df[df["algorithm"] == alg]

        for col_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx]

            for wid in word_ids:
                word_df = alg_df[alg_df["word_id"] == wid].copy()

                if word_df.empty:
                    continue

                word_df = word_df.set_index("config").reindex(config_order).reset_index()
                word_df = word_df.dropna(subset=[metric])

                if not word_df.empty:
                    word_label = word_df["word"].iloc[0] if "word" in word_df.columns else f"word_{wid}"
                    ax.plot(
                        word_df["config"],
                        word_df[metric],
                        marker='o',
                        label=f"{word_label}",
                        color=word_id_to_color[wid],
                        linewidth=1.5,
                        markersize=5
                    )

            ax.set_ylabel(label, fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45, labelsize=8)

            if row_idx == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(f"{alg.upper()}\n{label}", fontsize=10)

            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center right',
        bbox_to_anchor=(1.12, 0.5),
        title="Word",
        fontsize=8
    )

    fig.suptitle("Per-Word Performance Across Configurations", fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Per-word comparison graph saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main(base_dir: str | Path = None, output_path: str | Path = None):
    """
    Main function to generate overall analysis graphs.

    Args:
        base_dir: Base directory containing visualizations. Defaults to homograph_data/visualizations
        output_path: Where to save the output graph. Defaults to homograph_data/visualizations/overall_comparison.png
    """
    if base_dir is None:
        base_dir = Path(__file__).parent / "homograph_data" / "visualizations"

    if output_path is None:
        output_path = Path(base_dir) / "overall_comparison.png"

    print(f"Loading CSVs from: {base_dir}")
    df = load_all_results(base_dir)

    print(f"Loaded {len(df)} rows from {df['config'].nunique() if 'config' in df.columns else 'N/A'} configurations")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Layers: {df['layer'].unique().tolist()}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")

    create_comparison_graphs(df, output_path)


if __name__ == "__main__":
    ## get params
    import json
    from homograph_analysis.models import MODEL_PATHS
    from homograph_analysis import utils
    config_path = 'jabert_config.json'
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    VIS_PATH = Path(config['outfolder']) / "visualizations"

    print(f"Loading CSVs from: {VIS_PATH}")
    df = load_all_results(VIS_PATH)

    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Layers: {df['layer'].unique().tolist()}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")

    create_comparison_graphs(df, VIS_PATH / "overall_comparison.png")

    create_per_word_comparison_graphs(df, output_path=VIS_PATH / "per_word_comparison.png")
