"""
Scratch-script to get word-embeddings out of Bert-family Models.


:author: James V. Bruno
"""
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer, BertModel
from torch.nn.modules import Embedding
from torch import Tensor


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parents[1] / "data"
LAVA_FILE = DATA_DIR / "lava" / "all_verbs.csv"


def get_word_emebeddings(
    verb: str, embeddings: Embedding, tokenizer: BertTokenizer
) -> Tensor:
    """Return the embedding vector for ``verb``.

    If the tokenizer gives us sub-word pieces, we take the average of the
    embeddings for all the word-pieces, otherwise, we just return the vector.

    Parameters
    ----------
    verb : str
        Verb, as from the LAVA dataset.
    embeddings : Embedding
        The vocab_size x n_dimension embedding layer for the model.
    tokenizer : BertTokenizer
        The tokenizer for the model.

    Returns
    -------
    word_embedding : Tensor
        an 1 x n_dimension Tensor with the word-embeddings for ``verb``.
    """
    inputs = tokenizer(verb, add_special_tokens=False)
    input_ids = inputs["input_ids"]

    if len(input_ids) > 1:
        # e.g [21877, 28090], for "peddled" -> ['pe', '##ddled']
        # we have sub-word tokenization, so take the mean
        word_embedding = embeddings.weight[input_ids].mean(axis=0)

    else:
        # Just one token here, so take the first (and only) embedding so the
        # shape is the same as what we have above.
        word_embedding = embeddings.weight[input_ids][0]

    return word_embedding


def main():

    data_df = pd.read_csv(LAVA_FILE)

    model = BertModel.from_pretrained("bert-base-uncased")
    embedding_layer = model.get_input_embeddings()
    tokenizer = tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # not sure what the best data structure is to work with, so for now
    # we'll build up a csv:
    dict_for_df = {}

    for verb in data_df["verb"]:
        word_embedding = get_word_emebeddings(verb, embedding_layer, tokenizer)

        dict_for_df[verb] = word_embedding.tolist()

    output_df = pd.DataFrame.from_dict(dict_for_df, orient="index")


if __name__ == "__main__":
    main()
