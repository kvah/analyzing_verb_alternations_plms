# ling-575-analyzing-nn-group

Collaborators: David Yi, Jiayu Han, Jimmy Bruno, Peter Zukerman

This Repository contains the `alternation_prober` package and related resources.
`alternation_prober` is used in experiments for 575 "Analyzing Neural Language Models"
in Spring of 2022.

## Installation

1. Clone the repository: `git clone git@github.com:kvah/ling-575-analyzing-nn-group.git`

2. `cd` into the clone directory: `cd ling-575-analyzing-nn-group/`

3. Create the conda environment: `conda env create -n 575nn --file ./conda_environment.yaml`

5. Activate the conda environment: `conda activate 575nn`

6. Install `alternationprober` as an editable package with pip: `pip install -e .`

## Development and Tests
To Run tests: `pytests ./tests`


## `alternationprober` package.
Provides the following:

 *  `get_bert_word_embeddings`: command-line utility to produce word-level
    embeddings from the LAVA dataset.
     * usage: `get_bert_word_embeddings`
     * Will produce 2 output files:
         * `./data/embeddings/bert-word-embeddings-lava.npy`: This is a 2d
           numpy array with the word-embeddings
         * `./data/embeddings/bert-word-embeddings-lava-vocab.json`:  This is
           a mapping of vocabulary item to its index in the numpy array.
           (It is the same as the order from the original file in the LAVA
           dataset.)

The embeddings and associated vocabulary mapping can be loaded like so:

```
import json
import numpy as np

from alternationprober.constants import (PATH_TO_BERT_WORD_EMBEDDINGS_FILE,
                                         PATH_TO_LAVA_VOCAB)

embeddings = np.fromfile(PATH_TO_BERT_WORD_EMBEDDINGS_FILE)

with PATH_TO_LAVA_VOCAB.open("r") as f:
    vocabulary_to_index = json.load(f)
```


## Other Related Resources in This Repository

### `download-datasets.sh`
Shell script to download the LaVa and FAVA datasets to `./data` directory.

Usage: `sh download-datasets.sh`

### Proposal
latex source of our proposal


