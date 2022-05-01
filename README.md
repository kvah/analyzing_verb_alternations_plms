# ling-575-analyzing-nn-group

Collaborators: David Yi, Jiayu Han, Jimmy Bruno, Peter Zukerman

## Contents

### `download-datasets.sh`
Shell script to download the LaVa and FAVA datasets to `./data` directory.

Usage: `sh download-datasets.sh`

### Proposal
latex source of our proposal

### src/embeddings
  *  `get_embeddings.py`: script to produce word-level embeddings from the LAVA dataset.
     * usage: `python get_embeddings.py`
     * Will produce the output file `./data/embeddings/bert-embeddings-lava.csv"` (path hard-coded ¿for now?)


## Tests

To Run tests: `pytests ./tests`