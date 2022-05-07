"""
Tests for the embeddings.get_bert_word_embeddings module
"""
import json
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from torch import load
from torch.testing import assert_close
from transformers import BertTokenizer, BertModel
from unittest import TestCase, mock

from alternationprober.embeddings.get_bert_word_embeddings import (
    get_word_embeddings,
    main,
)

THIS_DIR = Path(__file__).resolve().parent
EXPECTED_OUTPUT_DIR = THIS_DIR / "expected_outputs"
INPUT_DIR = THIS_DIR / "test_inputs"


class TestGetWordEmbeddings(TestCase):
    """Test the get_word_embeddings() function."""

    @classmethod
    def setUpClass(cls):
        """Load the model we use for testing."""
        model = BertModel.from_pretrained("bert-base-uncased")
        cls.embedding_layer = model.get_input_embeddings()
        cls.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def check_dimension_size(self, computed_size, expected_size=768):

        if computed_size != expected_size:
            self.fail(
                f"embedding layer has {computed_size} dimensions. "
                f"Expected {expected_size}."
            )

    def test_no_sub_word_tokenization(self):
        """Test a case in which there is no sub-word tokenization.

        ``fed`` is such a case.
        """
        expected_output_file = EXPECTED_OUTPUT_DIR / "no_sub_word_tokenization.pt"
        expected_output = load(expected_output_file)

        verb = "fed"
        word_embedding = get_word_embeddings(verb, self.embedding_layer, self.tokenizer)

        self.check_dimension_size(len(word_embedding))

        assert_close(expected_output, word_embedding)

    def test_sub_word_tokenization(self):
        """Test a case in which the verb is tokenized into sub-word pieces.

        e.g [21877, 28090], for "peddled" -> ['pe', '##ddled'])
        """
        expected_output_file = EXPECTED_OUTPUT_DIR / "sub_word_tokenization.pt"
        expected_output = load(expected_output_file)

        verb = "peddled"
        word_embedding = get_word_embeddings(verb, self.embedding_layer, self.tokenizer)

        self.check_dimension_size(len(word_embedding))

        assert_close(expected_output, word_embedding)


class TestMain(TestCase):
    """Test the ``main()`` function in ``get_bert_word_embeddings.py``."""

    # Use this to mock the path constants:
    module_address = "alternationprober.embeddings.get_bert_word_embeddings"

    def test_main_success(self):
        """Test that main() works as expected when the lava dataset is present.

        We check to make sure that the embeddings were output correctly as npy files,
        and we check to make sure that we captured the mapping of vocabulary to indices.
        """
        expected_emeddings_file = EXPECTED_OUTPUT_DIR / "lava_test_embeddings.npy"
        expected_embeddings = np.load(expected_emeddings_file, allow_pickle=True)

        expected_vocabulary_file = EXPECTED_OUTPUT_DIR / "lava_test_vocab.json"
        expected_vocabulary = json.load(expected_vocabulary_file.open("r"))

        # Override the constant for the input file so we can point to our test file.
        input_path_to_mock = f"{self.module_address}.PATH_TO_LAVA_FILE"
        new_input_path = INPUT_DIR / "test_verbs.csv"

        # override the constant for the output embeddings file:
        output_embedding_path_to_mock = (
            f"{self.module_address}.PATH_TO_BERT_WORD_EMBEDDINGS_FILE"
        )

        # override the constant for the output vocabulary file:
        output_vocab_path_to_mock = f"{self.module_address}.PATH_TO_LAVA_VOCAB"

        with TemporaryDirectory() as tmp_dir:
            new_embedding_output_file = Path(tmp_dir) / "temp_embedding_output.npy"
            new_vocab_file = Path(tmp_dir) / "temp_vocab_output.json"

            with mock.patch(output_embedding_path_to_mock, new_embedding_output_file):
                with mock.patch(output_vocab_path_to_mock, new=new_vocab_file):
                    with mock.patch(input_path_to_mock, new=new_input_path):

                        main()

            # load in the embeddings file that should have been created and check it:
            computed_embeddings = np.load(new_embedding_output_file, allow_pickle=True)
            np.testing.assert_allclose(expected_embeddings, computed_embeddings)

            # load in the vocab file that should have been created and check it:
            computed_vocabulary = json.load(new_vocab_file.open("r"))
            self.assertDictEqual(expected_vocabulary, computed_vocabulary)

    def test_main_no_lava_file(self):
        """Test that Exception is raised if the lava dataset has not been downloaded.

        Make sure that we display a message about downloading the datasets
        if the Lava File does not exist.
        """
        # override the constant for the input file so we can make it not exist
        input_path_to_mock = f"{self.module_address}.PATH_TO_LAVA_FILE"

        # override the constant for the output file so we don't overwrite it.
        output_path_to_mock = f"{self.module_address}.PATH_TO_BERT_WORD_EMBEDDINGS_FILE"

        with TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "I-do-not-exist.file"
            output_file = Path(tmp_dir) / "temp_output.file"

            with mock.patch(input_path_to_mock, input_file):
                with mock.patch(output_path_to_mock, output_file):
                    with self.assertRaisesRegex(FileNotFoundError, "download-datasets"):
                        main()
