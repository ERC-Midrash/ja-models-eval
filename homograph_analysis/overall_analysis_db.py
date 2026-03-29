import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Any, Dict


def format_params_str(params: Any) -> str:
    """
    Format params dict/JSON as a compact string for display.

    Args:
        params: Dict or JSON string of parameters

    Returns:
        Compact string like "k=7, cosine"
    """
    if params is None:
        return ""

    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return params

    if not isinstance(params, dict):
        return str(params)

    parts = []
    for key, value in params.items():
        if key in ('k_neighbors', 'k'):
            parts.append(f"k={value}")
        elif key == 'distance':
            parts.append(str(value))
        elif key == 'linkage':
            parts.append(str(value))
        else:
            parts.append(f"{key}={value}")

    return ", ".join(parts)


def prepare_df_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame from the database for plotting.

    Takes a DataFrame from get_runs/get_latest and:
    - Parses the metrics JSON column into separate columns
    - Renames clustering_alg to algorithm for compatibility
    - Converts layer to string for display purposes

    Args:
        df: DataFrame from get_runs or get_latest

    Returns:
        DataFrame ready for plotting with expanded metrics columns
    """
    if df.empty:
        return df.copy()

    result = df.copy()

    if 'metrics' in result.columns:
        metrics_expanded = result['metrics'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        metrics_df = pd.json_normalize(metrics_expanded)
        result = pd.concat([result.drop(columns=['metrics']), metrics_df], axis=1)

    if 'clustering_alg' in result.columns:
        result = result.rename(columns={'clustering_alg': 'algorithm'})

    if 'model' in result.columns:
        result['model'] = result['model'].str.lower()

    if 'layer' in result.columns:
        result['layer'] = result['layer'].astype(str)

    if 'params' in result.columns:
        result['params_str'] = result['params'].apply(format_params_str)

    return result


def get_config_order(
        df: pd.DataFrame,
        models: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None
) -> List[str]:
    """
    Build configuration order for plotting by sorting actual configs from data.

    Args:
        df: Prepared DataFrame (must have 'model', 'layer', 'algorithm', 'config' columns)
        models: List of models in desired order. If None, extracts from DataFrame.
        layers: List of layers in desired order. If None, extracts from DataFrame.
        algorithms: List of algorithms in desired order. If None, extracts from DataFrame.

    Returns:
        List of config strings sorted by (model, layer, algorithm) order
    """
    if models is None:
        models = sorted(df['model'].unique().tolist())
    if layers is None:
        layers = sorted(df['layer'].unique().tolist(), key=lambda x: int(x) if x.lstrip('-').isdigit() else 0)
    if algorithms is None:
        algorithms = sorted(df['algorithm'].unique().tolist())

    model_order = {m: i for i, m in enumerate(models)}
    alg_order = {a: i for i, a in enumerate(algorithms)}

    configs = df['config'].unique().tolist()

    def sort_key(config: str):
        lines = config.split('\n')
        model = lines[0]
        alg = lines[1].split(' ')[0] if len(lines) > 1 else ''
        return (
            model_order.get(model, 999),
            alg_order.get(alg, 999)
        )

    return sorted(configs, key=sort_key)


def create_comparison_graphs(
        df: pd.DataFrame,
        output_path: Optional[str | Path] = None,
        models: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
        model_labels: Optional[Dict[str, str]] = None,
        n_colors:int = None  ### might want to control the number of colors for consistency
) -> None:
    """
    Create vertically stacked graphs comparing metrics across models/layers/algorithms.

    X-axis: (model, layer, algorithm) tuples
    Y-axis: percent_accurate, RAND_index (one per graph)

    Args:
        df: Prepared DataFrame (output of prepare_df_for_plotting)
        output_path: Path to save the figure. If None, displays instead.
        models: List of models in desired order. If None, extracts from DataFrame.
        layers: List of layers in desired order. If None, extracts from DataFrame.
        algorithms: List of algorithms in desired order. If None, extracts from DataFrame.
        model_labels: Dict mapping model name to display label for x-axis
                      (e.g. {'jabert': 'JABert', 'hearbert': 'HeBERT'}).
                      If None, uses model names as-is.
    """
    df = df.copy()

    if model_labels:
        df["config"] = df["model"].map(lambda m: model_labels.get(m, m))
    elif 'params_str' in df.columns:
        df["config"] = df["model"] + "\n" + df["algorithm"] + " " + df["params_str"]
    else:
        df["config"] = df["model"] + "\n" + df["algorithm"]

    config_order = get_config_order(df, models, layers, algorithms)

    if not config_order:
        print("No matching configurations found in data")
        return

    metrics = []
    metric_labels = []

    if 'percent_accurate' in df.columns:
        metrics.append('percent_accurate')
        metric_labels.append('Accuracy (%)')
    if 'RAND_index' in df.columns:
        metrics.append('RAND_index')
        metric_labels.append('Adjusted RAND Index')
    if 'silhouette' in df.columns:
        metrics.append('silhouette')
        metric_labels.append('Silhouette Score')

    if not metrics:
        print("No metrics columns found in DataFrame")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(8, len(config_order) * 1.2), 4 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    palette = sns.color_palette("husl", n_colors=n_colors or len(config_order))

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
        ax.set_ylabel(label, fontsize=14)
        ax.tick_params(axis='x', rotation=0, labelsize=11)

        n_per_group = len([c for c in config_order if c.split('\n')[0] == config_order[0].split('\n')[0]])
        for i in range(n_per_group - 1, len(config_order) - 1, n_per_group):
            ax.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.5)

    axes[0].set_title("Homograph Discrimination Performance", fontsize=16, fontweight='bold')
    axes[-1].set_xlabel("Configuration (Model / Flow)", fontsize=14)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def get_simple_config_order(
        df: pd.DataFrame,
        models: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        model_labels: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Build simple configuration order for per-word plots (model_L{layer} format).

    Args:
        df: Prepared DataFrame
        models: List of models in desired order. If None, extracts from DataFrame.
        layers: List of layers in desired order. If None, extracts from DataFrame.
        model_labels: Dict mapping model name to display label. If None, uses model names as-is.

    Returns:
        List of config strings in format "label_L{layer}"
    """
    if models is None:
        models = sorted(df['model'].unique().tolist())
    if layers is None:
        layers = sorted(df['layer'].unique().tolist(), key=lambda x: int(x) if x.lstrip('-').isdigit() else 0)

    config_order = []
    if model_labels:
        for model in models:
            config_order.append(model_labels.get(model, model))
    else:
        for model in models:
            for layer in layers:
                config_order.append(f"{model}_L{layer}")

    return config_order


def create_per_word_comparison_graphs(
        df: pd.DataFrame,
        word_ids: Optional[List[int]] = None,
        output_path: Optional[str | Path] = None,
        models: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        model_labels: Optional[Dict[str, str]] = None
) -> None:
    """
    Create plots showing metric values for each word_id across model & layer configurations.

    For each clustering method, creates a row of plots (one per metric).
    Each plot shows lines for each word_id, with x-axis being (model, layer).

    Args:
        df: Prepared DataFrame (output of prepare_df_for_plotting)
        word_ids: List of word_ids to include. If None, uses all word_ids in the data.
        output_path: Path to save the figure. If None, displays instead.
        models: List of models in desired order. If None, extracts from DataFrame.
        layers: List of layers in desired order. If None, extracts from DataFrame.
        model_labels: Dict mapping model name to display label for x-axis
                      (e.g. {'jabert': 'JABert', 'hearbert': 'HeBERT'}).
                      If None, uses model names as-is.
    """
    df = df.copy()
    if model_labels:
        df["config"] = df["model"].map(lambda m: model_labels.get(m, m))
    else:
        df["config"] = df["model"] + "_L" + df["layer"]

    config_order = get_simple_config_order(df, models, layers, model_labels)

    available_configs = set(df["config"].unique())
    config_order = [c for c in config_order if c in available_configs]

    if not config_order:
        print("No matching configurations found in data")
        return

    df = df[df["config"].isin(config_order)]

    if word_ids is None:
        word_ids = sorted(df["word_id"].unique().tolist())
    else:
        df = df[df["word_id"].isin(word_ids)]

    clustering_methods = sorted(df["algorithm"].unique().tolist())

    metrics = []
    metric_labels = []
    if 'n_mistakes' in df.columns:
        metrics.append('n_mistakes')
        metric_labels.append('Number of Mistakes')
    if 'percent_accurate' in df.columns:
        metrics.append('percent_accurate')
        metric_labels.append('Accuracy')
    if 'RAND_index' in df.columns:
        metrics.append('RAND_index')
        metric_labels.append('RAND Index')
    if 'silhouette' in df.columns:
        metrics.append('silhouette')
        metric_labels.append('Silhouette Score')

    if not metrics:
        print("No metrics columns found in DataFrame")
        return

    n_rows = len(clustering_methods)
    n_cols = len(metrics)

    if n_rows == 0:
        print("No clustering methods found in data")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    palette = sns.color_palette("husl", n_colors=len(word_ids))
    word_id_to_color = {wid: palette[i] for i, wid in enumerate(word_ids)}

    for row_idx, alg in enumerate(clustering_methods):
        alg_df = df[df["algorithm"] == alg]

        for col_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx][col_idx]

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

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
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


if __name__ == '__main__':
    from homograph_analysis import db_utils
    models = ['jabert', 'hearbert', 'mbert', 'camelbert_msa_sweet', 'camelbert_ca_sweet']
    # models = ['jabert', 'hearbert', 'mbert']
    tag = 'knn5'
    folder_name = 'sweet_samples'
    model_labels = {
        'jabert': 'JABERT',
        'hearbert': 'HeArBert',
        'mbert': 'mBERT',
        'camelbert_msa': 'Translit.\nCAMeLBERT MSA',
        'camelbert_msa_sweet': 'Translit.\nCAMeLBERT MSA\nSWEET',
        'camelbert_ca': 'Translit.\nCAMeLBERT CA',
        'camelbert_ca_sweet': 'Translit.\nCAMeLBERT CA\nSWEET',
    }
    layer = -1
    # alg = 'kmeans'
    # params = {'distance': 'cosine'}
    alg = 'knn'
    params = {'distance': 'cosine', 'k_neighbors': 5}
    # alg = 'silhouette_gt'
    # params = {'distance': 'cosine'}
    outfolder = f"homograph_analysis/homograph_data/{folder_name}/visualizations_fresh"

    outpath = outfolder + f'/overall_comparison_results_{tag}.png'
    # run_ids =  [288, 292, 296]
    run_ids = None
    if run_ids:
        latest_data = db_utils.get_runs(run_ids=run_ids)  # get_latest(models=models, clustering_alg=alg, layer=layer, params_filter=params)
    else:
        latest_data = db_utils.get_latest(models=models, clustering_alg=alg, layer=layer, params_filter=params)
    latest_data = prepare_df_for_plotting(latest_data)
    create_comparison_graphs(df=latest_data,
                             output_path=outpath,
                             models=models,
                             model_labels=model_labels,
                             n_colors=5)
    outpath = outfolder + "/overall_comparison_wordwise.png"
    create_per_word_comparison_graphs(df=latest_data,
                                      output_path=outpath,
                                      models=models,
                                      model_labels=model_labels)