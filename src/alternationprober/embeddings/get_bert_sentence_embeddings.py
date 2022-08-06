"""
Script to get sentence embeddings out of Bert-family Models.

This file is used in run_linear_classifier_sentence_experiment.py

Authors
-------
Jiayu Han (jyhan126@uw.edu)
David Yi (davidyi6@uw.edu)

Last Updated
-------
7/31/22

"""

import argparse 

from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from typing import List
import pandas as pd
import torch
import numpy as np

from alternationprober.constants import (
    PATH_TO_FAVA_DIR,
    PATH_TO_SENTENCE_EMBEDDINGS_DIR,
)

PATH_TO_SENTENCE_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
alternations = ['combined', 'dative', 'inchoative', 'spray_load', 'there', 'understood']

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


def get_model_embeddings(sents:List[str], model:AutoModel, tokenizer:AutoTokenizer) -> Tensor:
    """
    sents: a list of sentences e.g. [[christopher fed the casserole to the children .]
                                    [christopher fed the children the casserole .]]
    model: bert model
    tokenizer: corresponding tokenizer
    layers: a list: the layers in bert you want to probe, e.g. [0,1,2]
    return: average of these embeddings: in this example, it will average all embeddings of words in different layers
    """
    tokens_in_sent = tokenizer.batch_encode_plus(sents, pad_to_max_length=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens_in_sent)

        # hidden_states: tuple, len(hidden_states) = 13, each element in tuple represents its corresponding layer
        hidden_states = outputs[2][1:]

        #in each hidden_state, there is a tensor: e.g. [2 * 14 * 768], 2 means the number of sentences, 14: max_len in
        # these sentences, 768: the dimension of word embeddings
        # firstly, for each hidden layer, sum all words embeddings, after that, we can get a list of [13 * 2 * 768]
        full_embeddings = [torch.sum(hidden_state, dim=1) for hidden_state in hidden_states]

        #secondly, according to the designated layers, select these embeddings for these layers [3 * 8 * 768]
        return torch.stack(full_embeddings)

def get_sent_embeddings(path, model, tokenizer):
    fava_df = pd.read_csv(path, sep='\t+', names=['alternation', 'labels', 'sentence'])
    # load all sentences
    sentences = fava_df['sentence'].to_list()
    sents_embeddings = get_model_embeddings(sentences, model, tokenizer)
    # return numpy for future classification
    return sents_embeddings.detach().numpy()

def main():
    if '/' in args.model_name:
        model_name = args.model_name.split('/')[1]
    else:
        model_name = args.model_name
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for alternation in alternations:
        train_embeddings = get_sent_embeddings(PATH_TO_FAVA_DIR / alternation / 'train.tsv', model, tokenizer)
        dev_embeddings = get_sent_embeddings(PATH_TO_FAVA_DIR / alternation / 'dev.tsv', model, tokenizer)
        test_embeddings = get_sent_embeddings(PATH_TO_FAVA_DIR / alternation / 'test.tsv', model, tokenizer)


        
        model_embed_path = PATH_TO_SENTENCE_EMBEDDINGS_DIR / model_name / alternation
        model_embed_path.mkdir(parents=True, exist_ok=True)

        train_path = model_embed_path / 'train.npy' 
        np.save(train_path, train_embeddings)
        print(f'Sentence (train) embeddings saved to: {train_path} for {model_name} for {alternation}')

        dev_path = model_embed_path / 'dev.npy'
        np.save(dev_path, dev_embeddings)
        print(f'Sentence (dev) embeddings saved to: {dev_path} for {model_name} for {alternation}')

        test_path = model_embed_path / 'test.npy'
        np.save(test_path, test_embeddings)
        print(f'Sentence (test) embeddings saved to: {test_path} for {model_name} for {alternation}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract embeddings from specified pretrained language model")
    parser.add_argument(
        "--model_name", 
        type=str,
        choices=['bert-base-uncased', 'roberta-base', 'google/electra-base-discriminator', 'microsoft/deberta-base'], 
        default='bert-base-uncased'
    )
    args = parser.parse_args()
    main()
