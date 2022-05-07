"""
Module containing constants for the ``alternation_prober`` package.

Constants:
----------
    BERT_WORD_EMBEDDINGS_FILE : Path
        Path to word-embeddings from lava dataset as produced from ``get_embeddings.py``.
    LAVA_FILE : Path
        Path to original Lava file as downloaded from ``https://nyu-mll.github.io/CoLA/``
    PATH_TO_LAVA_VOCAB : Path
        Path to mapping file of Lava Vocabulary to index
"""
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parents[1] / "data"

PATH_TO_LAVA_DIR = _DATA_DIR / "lava"
PATH_TO_LAVA_FILE = PATH_TO_LAVA_DIR / "all_verbs.csv"

PATH_TO_FAVA_DIR = _DATA_DIR / "fava" / "verb_classes_public"
PATH_TO_BERT_WORD_EMBEDDINGS_FILE = (
    _DATA_DIR / "embeddings" / "bert-word-embeddings-lava.npy"
)
PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE = (
    _DATA_DIR / "embeddings" / "bert-context-word-embeddings.npy"
)

PATH_TO_LAVA_VOCAB = _DATA_DIR / "embeddings" / "bert-word-embeddings-lava-vocab.json"

PATH_TO_RESULTS_DIRECTORY = _THIS_DIR.parents[1] / "results"
