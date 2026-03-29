from pathlib import Path

from homograph_analysis import utils
import json

from homograph_analysis.homograph_dataset_processing import main as generate_embeddings, OVERWRITE, ABORT, EXISTING
from homograph_analysis.clusteration import cluster_all, agglomerative_cluster_all
from homograph_analysis.knn import knn_loocv_all
from homograph_analysis.embedding_topology_metrics import compute_silhouette
from homograph_analysis.models import MODEL_PATHS


def run_config(config_path, overwrite_data=None, sample_ids:list=None, outfolder=None):
    """
    run analysis for a given config
    :param config_path:
    :param overwrite_data:
    :param sample_ids: optional - limit to specific sample ids in the datasets. These only kick in after embedding generation, in the clustering
    :return:
    """
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    if outfolder: ## replace outfolder in config
        config['outfolder'] = outfolder

    model = config['model'].upper()
    MODEL_PATH = MODEL_PATHS[config['model'].upper()]
    layer = int(config['layer'])

    XLSX_PATH = config['dataset_xlsx']
    JSONL_PATH = utils.get_embedding_path(config_path)
    VIS_PATH = Path(config['outfolder']) / "visualizations_fresh"
    vis_output_dir = Path(VIS_PATH) / Path(JSONL_PATH).stem

    generate_embeddings(xlsx_path=XLSX_PATH, model_path=MODEL_PATH, output_path=JSONL_PATH,
                        layer=layer, clean_geresh=model.upper()=='GERESHLESS', instruction=overwrite_data)

    word_ids = list(range(1, 17))

    # K-means clustering
    cluster_all(JSONL_PATH, word_ids, output_dir=vis_output_dir,
                run_tag=f'_{model.lower()}_layer{abs(layer)}',
                sample_ids=sample_ids,
                model=model, layer=layer, distance='cosine')

    # # Agglomerative clustering
    # agglomerative_cluster_all(JSONL_PATH, word_ids, output_dir=vis_output_dir, sample_ids=sample_ids, linkage="ward",
    #                           run_tag=f'_{model.lower()}_layer{abs(layer)}', model=model, layer=layer)
    #
    num_neigh = 5

    knn_loocv_all(JSONL_PATH, word_ids, output_dir=vis_output_dir,
                  sample_ids=sample_ids,
                  n_neighbors=num_neigh, run_tag=f'_{model.lower()}_layer{abs(layer)}_n{num_neigh}',
                  distance='cosine', model=model, layer=layer
    )

    # Silhouette scores on GT labels
    compute_silhouette(JSONL_PATH, word_ids, output_dir=vis_output_dir, sample_ids=sample_ids,
        run_tag=f'_{model.lower()}_layer{abs(layer)}', distance='cosine',
        model=model, layer=layer
    )

    compute_silhouette(JSONL_PATH, word_ids, output_dir=vis_output_dir, sample_ids=sample_ids,
        run_tag=f'_{model.lower()}_layer{abs(layer)}', distance='euclid',
        model=model, layer=layer
    )


if __name__ == '__main__':

    DATA_TREATMENT = EXISTING

    ### no sample filtering
    # OUTFOLDER = None
    # sample_ids = None

    ### with sample filtering (edit OUTFOLDER and sample_ids for your setup)
    OUTFOLDER = "homograph_analysis/homograph_data/sweet_samples"
    sample_ids = None  # Set to a list of integer IDs to filter, or None for all samples


    ## get params
    run_config('jabert_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
    run_config('hearbert_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
    run_config('mbert_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
    run_config('camelbert_ca_sweet_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
    run_config('camelbert_ca_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
    run_config('camelbert_msa_sweet_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
    run_config('camelbert_msa_config.json', overwrite_data=DATA_TREATMENT, outfolder=OUTFOLDER, sample_ids=sample_ids)
