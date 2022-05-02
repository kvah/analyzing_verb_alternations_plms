"""
Script to get contextual word-embeddings out of Bert-family Models.

Contextual Word-embeddings from bert-base-uncased for the LaVa dataset
Input sentences used to create contextual embeddings are from FAVA
will be written to ``PATH_TO_BERT_WORD_EMBEDDINGS_CONTEXT_FILE`` as an
(|V|, 12, 768) dimensional nd-array.

Load the file with:

```
import numpy as np
from alternationprober.constants import PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE

embeddings = np.load(PATH_TO_BERT_WORD_EMBEDDINGS_FILE)
```

:author: David Yi
:date: 5/1/2022
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
import torch
from tqdm import tqdm

from alternationprober.constants import (
    PATH_TO_BERT_WORD_CONTEXT_EMBEDDINGS_FILE,
    PATH_TO_LAVA_FILE,
    PATH_TO_FAVA_DIR
)

PATH_TO_BERT_WORD_CONTEXT_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
EMBEDDING_SIZE = 768
NUM_LAYERS = 12

def get_sentences(verb:str) -> List[str]:
    """
    Returns all sentences from FAVA containing the input verb

    Parameters
    ----------
    verb : str
        Verb, as from the LAVA dataset.

    Returns
    -------
    sentences : List[str]
        The list of sentences from FAVA containing the input verb
    """ 
    # Regex string to check whether input verb exists in sentence
    contains_verb = fava_df[fava_df['sentence'].str.contains(rf'.*\s{verb}\s.*')]
    sentences = contains_verb['sentence'].to_list()
    return sentences

def find_verb_indices(verb_ids: Tensor, token_ids: Tensor) -> List[int]:
    """
    Get position of wordpiece tokens corresponding to the input verb in a sentence

    Parameters
    ----------
    verb_ids : Tensor
        1-d Tensor containing BERT wordpiece tokens corresponding to an input verb
    token_ids: Tensor
        1-d Tensor containing BERT wordpiece tokens corresponding to an input sentence

    Returns
    -------
    span : List[int]
        The index positions of the input verb relative to the sentence tokens
    """
    for i in range(0, len(token_ids)-2):
        if torch.equal(verb_ids, token_ids[i:i+len(verb_ids)]):
            span = [i, i+len(verb_ids)]
            return span

def get_verb_embedding(verb:str) -> Tensor: 
    """
    """
    inputs = tokenizer(verb_to_sentences[verb], padding=True, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)

    verb_ids = tokenizer(verb, padding=False)['input_ids'][1:-1]
    verb_ids = torch.tensor(verb_ids)
    # Get position of wordpiece tokens corresponding to the verb
    verb_positions = [find_verb_indices(verb_ids, token_ids) for token_ids in inputs['input_ids']]

    # The hidden state at index 0 is the output embedding 
    hidden_states = outputs['hidden_states'][1:]
    
    mean_embeddings = torch.empty(0, EMBEDDING_SIZE)

    for layer_idx in range(len(hidden_states)):
        layer_embedding = hidden_states[layer_idx]
        # Mean embedding for wordpiece tokens corresponding to the verb
        mean_embedding = torch.zeros(layer_embedding.shape[2])
        # Get average embedding over all sentences
        for sent_idx in range(layer_embedding.shape[0]):
            word_embedding = layer_embedding[sent_idx][verb_positions[sent_idx], :].mean(axis=0)
            mean_embedding += word_embedding 

        mean_embedding /= layer_embedding.shape[0]
        mean_embedding = mean_embedding.unsqueeze(0)
        mean_embeddings = torch.cat((mean_embeddings, mean_embedding))
    
    return mean_embeddings

if __name__ == "__main__":

    # Load LaVa
    try:
        lava_df = pd.read_csv(PATH_TO_LAVA_FILE)
    except FileNotFoundError as e:
        message = f"{PATH_TO_LAVA_FILE} not found.  Execute 'sh ./download-datasets.sh' before continuing."
        raise FileNotFoundError(message) from e
    
    # Load FAVA
    try:
        fava_train_path = PATH_TO_FAVA_DIR / "combined" / "train.tsv"
        # We only care about the sentences in dev/test, not the labels
        fava_dev_path = PATH_TO_FAVA_DIR / "combined" / "dev.tsv"
        fava_test_path = PATH_TO_FAVA_DIR / "combined" / "test.tsv"
        fava_train_df = pd.read_csv(fava_train_path, sep ='\t+', names=['alternation', 'label', 'sentence'])
        fava_dev_df = pd.read_csv(fava_dev_path, sep ='\t+', names=['alternation', 'label', 'sentence'])
        fava_test_df = pd.read_csv(fava_test_path, sep ='\t+', names=['alternation', 'label', 'sentence'])
        fava_df = pd.concat((fava_train_df, fava_dev_df, fava_test_df))
    except FileNotFoundError as e:
        message = f"{PATH_TO_FAVA_DIR} not found.  Execute 'sh ./download-datasets.sh' before continuing."
        raise FileNotFoundError(message) from e


    verbs = lava_df['verb']
    verb_to_sentences = {verb: get_sentences(verb) for verb in verbs}
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print(f'Creating contextual embeddings for {len(verbs)} verbs')
    # Shape: (|V|, 12, 768)
    verb_embeddings = torch.empty(verbs.shape[0], NUM_LAYERS, EMBEDDING_SIZE)
    for verb in tqdm(verbs):
        verb_embedding = get_verb_embedding(verb)
        verb_embedding = verb_embedding.unsqueeze(0)
        verb_embeddings = torch.cat((verb_embeddings, verb_embedding))

    verb_embeddings = verb_embeddings.detach().numpy()
    np.save(PATH_TO_BERT_WORD_CONTEXT_EMBEDDINGS_FILE, verb_embeddings)
