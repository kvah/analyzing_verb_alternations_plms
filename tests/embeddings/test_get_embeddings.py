"""
Tests for the embeddings.get_bert_word_embeddings module
"""
import json
from numpy import array
from numpy.testing import assert_allclose
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import BertTokenizer, BertModel
from unittest import SkipTest, TestCase, mock

from alternation_prober.constants import PATH_TO_LAVA_FILE
from alternation_prober.embeddings.get_bert_word_embeddings import (get_word_emebeddings,
                                                                    main)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "expected_outputs"


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
        expected_output_file = DATA_DIR / "no_sub_word_tokenization.json"

        with expected_output_file.open("r") as f:
            expected_output_as_np_array = array(json.load(f))

        verb = "fed"
        word_embedding = get_word_emebeddings(
            verb, self.embedding_layer, self.tokenizer
        )

        self.check_dimension_size(len(word_embedding))

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
        word_embedding = get_word_emebeddings(
            verb, self.embedding_layer, self.tokenizer
        )

        self.check_dimension_size(len(word_embedding))

        word_embedding_as_np_array = word_embedding.detach().numpy()

        assert_allclose(expected_output_as_np_array, word_embedding_as_np_array)


class TestMain(TestCase):
    """Test the ``main()`` function in ``get_bert_word_embeddings.py``."""

    # used to mock the path constants:
    module_address = "alternation_prober.embeddings.get_bert_word_embeddings"

    def test_main_success(self):
        """Test that main() works as expected when the lava dataset is present."""
        if not PATH_TO_LAVA_FILE.exists():
            raise SkipTest(f"{PATH_TO_LAVA_FILE} does not exist. "
                           "Please run 'sh ./download-datasets.sh'.")

        with TemporaryDirectory() as tmp_dir:
            # override the constant for the output file so we don't overwrite it:
            path_to_mock = f"{self.module_address}.PATH_TO_BERT_WORD_EMBEDDINGS_FILE"

            output_file = Path(tmp_dir) / "temp_output.file"
            with mock.patch(path_to_mock, output_file):
                main()

                # test that we generated an output file:
                if not output_file.exists():
                    self.fail("Failed to generate output file.")

    def test_main_no_lava_file(self):
        """Exception raised if the lava dataset has not been downloaded.

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
