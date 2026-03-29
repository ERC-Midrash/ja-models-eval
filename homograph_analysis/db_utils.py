import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


DEFAULT_DB_PATH = Path(__file__).parent / "homograph.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.

    Args:
        db_path: Path to the database file. If None, uses the default path.

    Returns:
        sqlite3.Connection with UTF-8 support
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA encoding = 'UTF-8'")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize the database with the required tables.

    Creates the following tables if they don't exist:
    - runs: stores metadata for each clustering run
    - run_results: stores per-word results for each run

    Args:
        db_path: Path to the database file. If None, uses the default path.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clustering_alg TEXT COLLATE NOCASE NOT NULL,
            model TEXT COLLATE NOCASE NOT NULL,
            layer INTEGER NOT NULL,
            params TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS run_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            word_id INTEGER NOT NULL,
            k INTEGER NOT NULL,
            n_samples INTEGER NOT NULL,
            gt_clusters TEXT NOT NULL,
            pred_clusters TEXT NOT NULL,
            metrics TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs (id)
        )
    """)

    conn.commit()
    conn.close()


def save_cluster_run(
        clustering_alg: str,
        model: str,
        layer: int,
        params: Dict[str, Any],
        results: List[Dict[str, Any]],
        db_path: Optional[Path] = None
) -> int:
    """
    Save a clustering run and its results to the database.

    Creates an entry in the `runs` table and corresponding entries in `run_results`.

    Args:
        clustering_alg: The clustering algorithm used (e.g., 'kmeans', 'agglomerative', 'knn')
        model: The model name (e.g., 'jabert', 'rambert')
        layer: The layer number
        params: Dict of algorithm parameters (e.g., {'distance': 'euclid', 'k': 2, 'linkage': 'ward'})
        results: List of result dictionaries, each containing:
            - word: The word text
            - word_id: The word ID
            - k: Number of clusters/neighbors
            - n_samples: Number of samples
            - gt_clusters: Ground truth clusters (string)
            - pred_clusters: Predicted clusters (string)
            - metrics: Dict with metric names and values (e.g., n_mistakes, percent_accurate, RAND_index)
        db_path: Path to the database file. If None, uses the default path.

    Returns:
        The ID of the created run entry
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    params_json = json.dumps(params) if not isinstance(params, str) else params

    cursor.execute(
        "INSERT INTO runs (clustering_alg, model, layer, params) VALUES (?, ?, ?, ?)",
        (clustering_alg, model, layer, params_json)
    )
    run_id = cursor.lastrowid

    for result in results:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, str):
            metrics = json.dumps(metrics)

        cursor.execute(
            """INSERT INTO run_results
               (run_id, word, word_id, k, n_samples, gt_clusters, pred_clusters, metrics)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                result["word"],
                result["word_id"],
                result["k"],
                result["n_samples"],
                result["gt_clusters"],
                result["pred_clusters"],
                metrics
            )
        )

    conn.commit()
    conn.close()

    return run_id


def get_runs(run_ids: List[int], db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get the data for specified runs as a DataFrame.

    Returns the content of run_results entries for the given run_ids,
    joined with run metadata (clustering_alg, model, layer, params).

    Args:
        run_ids: List of run IDs to retrieve
        db_path: Path to the database file. If None, uses the default path.

    Returns:
        DataFrame with columns: run_id, clustering_alg, model, layer, params,
        word, word_id, k, n_samples, gt_clusters, pred_clusters, metrics
    """
    if not run_ids:
        return pd.DataFrame()

    conn = get_connection(db_path)

    placeholders = ','.join('?' * len(run_ids))
    query = f"""
        SELECT
            r.id as run_id,
            r.clustering_alg,
            r.model,
            r.layer,
            r.params,
            r.created_at,
            rr.word,
            rr.word_id,
            rr.k,
            rr.n_samples,
            rr.gt_clusters,
            rr.pred_clusters,
            rr.metrics
        FROM run_results rr
        JOIN runs r ON rr.run_id = r.id
        WHERE r.id IN ({placeholders})
        ORDER BY r.id, rr.word_id
    """

    cursor = conn.cursor()
    cursor.execute(query, run_ids)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    data = []
    for row in rows:
        data.append({
            'run_id': row['run_id'],
            'clustering_alg': row['clustering_alg'],
            'model': row['model'],
            'layer': row['layer'],
            'params': row['params'],
            'created_at': row['created_at'],
            'word': row['word'],
            'word_id': row['word_id'],
            'k': row['k'],
            'n_samples': row['n_samples'],
            'gt_clusters': row['gt_clusters'],
            'pred_clusters': row['pred_clusters'],
            'metrics': row['metrics']
        })

    return pd.DataFrame(data)


def get_latest(
        models: List[str],
        clustering_alg: str,
        layer: int = -1,
        params_filter: Optional[Dict[str, Any]] = None,
        db_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Get the latest run data for specified models, clustering algorithm, and layer.

    Args:
        models: List of model names to retrieve
        clustering_alg: The clustering algorithm to filter by
        layer: Layer number to filter by. Defaults to -1.
        params_filter: Optional dict of param key-value pairs to filter by
                       e.g., {'distance': 'cosine'} or {'distance': 'euclid', 'linkage': 'ward'}
        db_path: Path to the database file.

    Returns:
        DataFrame with the latest run data for each model
    """
    if not models:
        return pd.DataFrame()

    conn = get_connection(db_path)
    cursor = conn.cursor()



    # Build the params filter clause
    params_conditions = ""
    params_values = []
    if params_filter:
        conditions = []
        for key, value in params_filter.items():
            conditions.append(f"json_extract(params, '$.{key}') = ?")
            params_values.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
        params_conditions = " AND " + " AND ".join(conditions)

    model_placeholders = ','.join('?' * len(models))

    query = f"""
        SELECT r.id
        FROM runs r
        INNER JOIN (
            SELECT model, MAX(created_at) as max_created
            FROM runs
            WHERE model IN ({model_placeholders}) 
              AND clustering_alg = ? 
              AND layer = ?
              {params_conditions}
            GROUP BY model
        ) latest ON r.model = latest.model AND r.created_at = latest.max_created
        WHERE r.clustering_alg = ? AND r.layer = ? {params_conditions}
    """

    # Build the full parameter list
    query_params = models + [clustering_alg, layer] + params_values + [clustering_alg, layer] + params_values

    cursor.execute(query, query_params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    run_ids = [row['id'] for row in rows]
    print(f"using these runs: {run_ids}")
    return get_runs(run_ids, db_path)