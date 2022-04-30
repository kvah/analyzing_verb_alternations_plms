#!/bin/sh
# script to download and extract The Lava and Fava datasets.
# see https://nyu-mll.github.io/CoLA/#lava
set -eu

# make some data directories if they're not already there
mkdir -p ./data/lava
mkdir -p .data/fava

# download and unzip the lava dataset:
wget https://nyu-mll.github.io/CoLA/lava.zip
unzip lava.zip all_verbs.csv -d data/lava

# downloand and unzip the fava dataset
wget https://nyu-mll.github.io/CoLA/fava.zip
unzip fava.zip -d data/fava

# cleanup
rm -f lava.zip
rm -f fava.zip
