import os

MODEL_PATHS = {
    'JABERT': os.environ.get('JABERT_MODEL_PATH', 'path/to/local/JABERT'),  # Set JABERT_MODEL_PATH env var or edit this
    'HEARBERT': 'aviadrom/HeArBERT',
    'MBERT': 'google-bert/bert-base-multilingual-cased',
    'BEREL': 'dicta-il/BEREL',
    'DICTA_BERT': r'dicta-il/dictabert',
    'ALEPHBERT': 'onlplab/alephbert-base',
    'HEBERT': 'avichr/heBERT',
    'CAMELBERT_CA_SWEET': 'CAMeL-Lab/bert-base-arabic-camelbert-ca',
    'CAMELBERT_MSA_SWEET': 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
}
