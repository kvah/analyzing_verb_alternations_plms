#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda env create -n 575nn --file ~/ling-575-analyzing-nn-group/conda_environment.yaml
conda activate 575nn
pip install -e ~/ling-575-analyzing-nn-group

python get_bert_context_word_embeddings.py
