# analyzing_verb_alternation_plms

## Installation

1. Clone the repository: `git clone git@github.com:kvah/analyzing_verb_alternations_plms.git`

2. `cd` into the clone directory: `cd analyzing_verb_alternations_plms/`

3. Create the conda environment: `conda env create -n 575nn --file ./conda_environment.yaml`

5. Activate the conda environment: `conda activate 575nn`

6. Install `alternationprober` as an editable package with pip: `pip install -e .`

## Development and Tests
To Run tests: `pytests ./tests`


## `alternationprober` package.
Provides the following:

 *  `get_bert_word_embeddings`: command-line utility to produce static word-level
    embeddings from the LAVA dataset.
     * usage: `get_bert_word_embeddings [--model_name]`
     * model_name: `bert-base-uncased`, `roberta-base`, `google/electra-base-discriminator`, `microsoft/deberta-base`
     * Will produce 2 output files:
         * `./data/embeddings/static/{model_name}.npy`: This is a 2d
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

embeddings = np.load(PATH_TO_BERT_WORD_EMBEDDINGS_FILE, allow_pickle=True))

with PATH_TO_LAVA_VOCAB.open("r") as f:
    vocabulary_to_index = json.load(f)
```

 *  `get_bert_word_embeddings`: command-line utility to produce contextual layer
    embeddings from the LAVA dataset.
     * usage: `get_bert_context_word_embeddings [--model_name]`
     * model_name: `bert-base-uncased`, `roberta-base`, `google/electra-base-discriminator`, `microsoft/deberta-base`
     * Produces the following output file:
         * `./data/embeddings/context/{model_name}.npy`: This is a 2d
           numpy array with the contextual word embeddings


The embeddings and associated vocabulary mapping can be loaded like so:

```
import json
import numpy as np

from alternationprober.constants import PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE

context_embeddings = np.load(PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE)
```

 *  `run_linear_classifier_experiment`: Will run our experiment to predict
    alternation classes from static model embeddings derived from the LaVA dataset.

     * usage: `run_linear_classifier_experiment [--output_directory] [--use_context_embeddings] [--model_name]`
     * Example: To run the linear classifier experiment with contextual embeddings:
        * `run_linear_classifier_experiment --use_context_embeddings --bert_base_uncased`
     * Note: <`output_directory`> will default to `./results/linear-probe-for-word-embeddings`
     * Note: `./download-datasets.sh` and the jupyter notebook `./data_analysis.ipynb` must be
             run first to make the data available.
             
 *  `run_linear_classifier_sentence_experiment`: Will run our experiment to predict
    sentence grammaticality from contextual embeddings derived from the FAVA dataset.

     * usage: `run_linear_classifier_experiment [--output_directory] [--use_context_embeddings] [--model_name]`
     * Example: To run the linear classifier experiment with contextual embeddings:
        * `run_linear_classifier_experiment --use_context_embeddings --bert_base_uncased`
     * Note: <`output_directory`> will default to `./results/linear-probe-for-sentence-embeddings`
     * Note: `./download-datasets.sh` and the jupyter notebook `./data_analysis.ipynb` must be
             run first to make the data available.
             

## Other Related Resources in This Repository

### `download-datasets.sh`
Shell script to download the LaVa and FAVA datasets to `./data` directory.

Usage: `sh download-datasets.sh`
