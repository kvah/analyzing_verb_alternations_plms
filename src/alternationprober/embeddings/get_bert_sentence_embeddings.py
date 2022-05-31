"""
Script to get sentence embeddings out of Bert-family Models.

This file is used in run_linear_classifier_sentence_experiment.py

:author: Jiayu Han
:date: 5/31/2022
"""

from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from typing import List
import pandas as pd
import torch


choices = ['combined', 'dative', 'inchoative', 'spray_load', 'there', 'understood']

def get_bert_embeddings(sents:List[str], model:AutoModel, tokenizer:AutoTokenizer, layers:List) -> Tensor:
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
        hidden_states = outputs[2]

        #in each hidden_state, there is a tensor: e.g. [2 * 14 * 768], 2 means the number of sentences, 14: max_len in
        # these sentences, 768: the dimension of word embeddings
        # firstly, for each hidden layer, sum all words embeddings, after that, we can get a list of [13 * 2 * 768]
        full_embeddings = [torch.sum(hidden_state, dim=1) for hidden_state in hidden_states]

        #secondly, according to the designated layers, select these embeddings for these layers [3 * 8 * 768]
        sents_embeddings = [full_embeddings[i] for i in layers]

        #finally, average the sentence embeddings by number of layers
        return sum(sents_embeddings)/len(sents_embeddings)

def get_sent_embeddings_dataset(path, layers):
    toy = pd.read_csv(path, sep='\t+', names=['alternation', 'labels', 'sentence'])
    labels = toy['labels'].to_list()
    # load all sentences
    sentences = toy['sentence'].to_list()
    model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sents_embeddings = get_bert_embeddings(sentences, model, tokenizer, layers=layers)
    # return numpy for future classification
    return [sents_embeddings.detach().numpy(), labels]

if __name__ == "__main__":
    path = '../../../data/fava/verb_classes_public/dative/toy.tsv'
    train_data = get_sent_embeddings_dataset(path, [12])