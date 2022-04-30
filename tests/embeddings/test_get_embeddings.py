"""
Tests for the embeddings.get_embeddings module
"""
import json
import sys
from numpy import array
from numpy.testing import assert_allclose
from pathlib import Path
from transformers import BertTokenizer, BertModel
from unittest import TestCase


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "expected_outputs"

MODULE_DIR = THIS_DIR.parents[1] / "src" / "embeddings"

sys.path.append(str(MODULE_DIR))
from get_embeddings import get_word_emebeddings


class TestGetWordEmbeddings(TestCase):
    """Test the get_word_embeddings() function."""

    @classmethod
    def setUpClass(self):
        """Load the model we use for testing."""
        model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding_layer = model.get_input_embeddings()
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def test_no_sub_word_tokenization(self):
        """Test a case in which there is no sub-word tokenization.

        ``fed`` is such a case.
        """
        expected_output_file = DATA_DIR / "no_sub_word_tokenization.json"

        with expected_output_file.open("r") as f:
            expected_output_as_np_array = array(json.load(f))

        verb = "fed"
        word_embedding = get_word_emebeddings(verb, self.embedding_layer, self.tokenizer)

        if len(word_embedding) != 768:
            self.fail(f"embedding layer has {len(get_word_embeddings)} dimensions. "
                      "Expected 768.")

        word_embedding_as_np_array = word_embedding.detach().numpy()

        assert_allclose(expected_output_as_np_array, word_embedding_as_np_array)


    def test_sub_word_tokenization(self):
        """Test a case in which the verb is tokenized into sub-word pieces.

        e.g [21877, 28090], for "peddled" -> ['pe', '##ddled'])
        """
        expected_output_file = DATA_DIR / "sub_word_tokenization.json"

        with expected_output_file.open("r") as f:
            expected_output_as_np_array = array(json.load(f))

        verb = "peddled"
        word_embedding = get_word_emebeddings(verb, self.embedding_layer, self.tokenizer)

        if len(word_embedding) != 768:
            self.fail(f"embedding layer has {len(get_word_embeddings)} dimensions. "
                      "Expected 768.")

        word_embedding_as_np_array = word_embedding.detach().numpy()

        assert_allclose(expected_output_as_np_array, word_embedding_as_np_array)
