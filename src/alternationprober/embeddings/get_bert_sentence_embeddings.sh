#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 575nn

# BERT 
python get_bert_sentence_embeddings.py \
    --model_name bert-base-uncased

# RoBERTa
python get_bert_sentence_embeddings.py \
    --model_name roberta-base

# ELECTRA
python get_bert_sentence_embeddings.py \
    --model_name google/electra-base-discriminator

# DeBERTa
python get_bert_sentence_embeddings.py \
    --model_name microsoft/deberta-base
