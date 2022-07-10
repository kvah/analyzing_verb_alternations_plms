"""
Script to get contextual word-embeddings out of Bert-family Models.

Contextual Word-embeddings from bert-base-uncased for the LaVa dataset
Input sentences used to create contextual embeddings are from FAVA
will be written to ``PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE`` as an
(|V|, 12, 768) dimensional nd-array.

Load the file with:

```
import numpy as np
from alternationprober.constants import PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE

embeddings = np.load(PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE)
```

:author: David Yi
:date: 5/1/2022
"""

import sys 
import argparse 

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
import torch
from tqdm import tqdm

from alternationprober.constants import (
    PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE,
    PATH_TO_LAVA_FILE,
    PATH_TO_FAVA_DIR
)

PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# These models happen to have the number of layers and embedding shapes, but this is not always true
MODEL_PARAMS = {
    'bert-base-uncased': {
        'num_layers': 12,
        'embedding_size': 768
    },
    'roberta-base': {
        'num_layers': 12,
        'embedding_size': 768
    },
    'google/electra-base-discriminator': {
        'num_layers': 12,
        'embedding_size': 768
    },
    'microsoft/deberta-base': {
        'num_layers': 12,
        'embedding_size': 768
    },
}

def get_sentences(verb:str, sentence_df:pd.DataFrame) -> List[str]:
    """
    Returns all sentences from FAVA containing the input verb

    Parameters
    ----------
    verb : str
        Verb, as from the LAVA dataset.
    sentence_df: pd.DataFrame
        pandas DataFrame containing sentences from FAVA

    Returns
    -------
    sentences : List[str]
        The list of grammatical sentences from FAVA containing the input verb.
    """ 
    # Mask to check whether input verb exists in sentence using regex.
    contains_verb_mask = sentence_df['sentence'].str.contains(rf'.*\s{verb}\s.*')

    # Mask to check whether sentence is grammatical
    grammatical_mask = sentence_df['label'] == 1
    combined_mask = contains_verb_mask & grammatical_mask
    sentences = sentence_df[combined_mask]['sentence'].to_list()

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

def get_verb_embedding(verb:str, verb_to_sentences:Dict, model:AutoModel, tokenizer:AutoTokenizer) -> Tensor: 
    """
    """
    inputs = tokenizer(verb_to_sentences[verb], padding=True, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    sentences = verb_to_sentences[verb]

    # Get position of wordpiece tokens corresponding to the verb
    verb_positions = get_indices(verb, sentences, inputs)

    # The hidden state at index 0 is the output embedding 
    hidden_states = outputs['hidden_states'][1:]
    
    mean_embeddings = torch.empty(0, hidden_states[0].shape[-1])

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

def get_indices(word, sentences, encoded_inputs):

    all_indices = []
    for i in range(len(sentences)):

        sentence = sentences[i]
        word_to_indices = []
        for word_id in encoded_inputs.word_ids(i):
            if word_id is not None:
                start, end = encoded_inputs[i].word_to_tokens(word_id)
                if start == end - 1:
                    tokens = [start]
                else:
                    tokens = [start, end-1]
                if len(word_to_indices) == 0 or word_to_indices[-1] != tokens:
                    word_to_indices.append(tokens)
        
        tokens = sentence.split(' ')
        if len(word.split(' ')) == 1:
            word_pos = tokens.index(word)
            verb_indices = word_to_indices[word_pos]
        else:
            first_word = word.split(' ')[0]
            first_index = tokens.index(first_word)

            word_pos = word_to_indices[first_index:first_index+len(word.split(' '))]
            verb_indices = [x for xs in word_pos for x in xs]

        all_indices.append(verb_indices)

    return all_indices

def main():
    """
    Extract word-level contextual embeddings from ``bert-base-uncased`` for the verbs in lava
    using the sentences in FAVA as input sentences
    """
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
    verb_to_sentences = {
        verb: get_sentences(verb=verb, sentence_df=fava_df) 
        for verb in verbs
    }

    model_name = args.model_name
    model_params = MODEL_PARAMS[model_name]
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f'Creating contextual embeddings for {len(verbs)} verbs')
    # Shape: (|V|, 12, 768)
    verb_embeddings = torch.empty(0, model_params['num_layers'], model_params['embedding_size'])

    for verb in tqdm(verbs):
        verb_embedding = get_verb_embedding(
            verb=verb, 
            verb_to_sentences=verb_to_sentences, 
            model=model, 
            tokenizer=tokenizer
        )
        verb_embedding = verb_embedding.unsqueeze(0)
        verb_embeddings = torch.cat((verb_embeddings, verb_embedding))

    verb_embeddings = verb_embeddings.detach().numpy()
    np.save(PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE, verb_embeddings)
    print(f'Context embeddings saved to: {PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE} for {model_name}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract embeddings from specified pretrained language model")
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    args = parser.parse_args(sys.argv[1:])

    main()
