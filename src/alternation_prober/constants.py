"""
Module containing constants for the ``alternation_prober`` package.

Constants:
----------
    BERT_WORD_EMBEDDINGS_FILE : Path
        Path to word-embeddings from lava dataset as produced from ``get_embeddings.py``.
    LAVA_FILE : Path
        Path to original Lava file as downloaded from ``https://nyu-mll.github.io/CoLA/``
"""
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parents[1] / "data"

PATH_TO_LAVA_FILE = _DATA_DIR / "lava" / "all_verbs.csv"
PATH_TO_BERT_WORD_EMBEDDINGS_FILE = (
    _DATA_DIR / "embeddings" / "bert-word-embeddings-lava.csv"
)
