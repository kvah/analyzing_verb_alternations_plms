#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
# conda env create -n 575nn --file ~/ling-575-analyzing-nn-group/conda_environment.yaml
conda activate 575nn
# pip install -e ~/ling-575-analyzing-verb-alternations-BERT

# # BERT 
# python get_bert_context_word_embeddings.py \
#     --model_name bert-base-uncased

# # RoBERTa
# python get_bert_context_word_embeddings.py \
#     --model_name roberta-base

# ELECTRA
python get_bert_context_word_embeddings.py \
    --model_name google/electra-base-discriminator

# DeBERTa
python get_bert_context_word_embeddings.py \
    --model_name microsoft/deberta-base
