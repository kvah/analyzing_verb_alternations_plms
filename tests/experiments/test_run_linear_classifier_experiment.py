"""
Tests for the experiments.run_linear_classifier_experiment module
"""
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
from tempfile import TemporaryDirectory


from alternationprober.experiments.run_linear_classifier_experiment import (
    get_evaluation_df,
    run_experiment_for_alternation_df,
)

from alternationprober.constants import (
    PATH_TO_BERT_WORD_EMBEDDINGS_FILE,
    PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE,
)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

# TODO: I think existence tests are sufficient for now, since the target predictions are subject to change as/if 
# we change our methodology

# def test_get_evaluation_df():
#     """Test the get_evaluation_df() function."""
#     path_to_expected_output_df = DATA_DIR / "labelled_data_evaluation_metrics.csv"
#     expected_output_df = pd.read_csv(path_to_expected_output_df)

#     path_to_predictions_df = DATA_DIR / "labelled_data_predictions.csv"
#     predictions_df = pd.read_csv(path_to_predictions_df)

#     alternation_classes = ["subclass1", "subclass2", "subclass3"]

#     computed_output_df = get_evaluation_df(predictions_df, alternation_classes)

#     assert_frame_equal(expected_output_df, computed_output_df)


def test_run_experiment_for_alternation_df_static():
    """Test the run_experiment_for_alternation_csv() function for static embeddings."""

    path_to_input_csv = DATA_DIR / "labelled_data.csv"
    # path_to_expected_predictions_csv = DATA_DIR / "labelled_data_predictions.csv"
    # expected_predictions_df = pd.read_csv(path_to_expected_predictions_csv, index_col='verb')

    word_embeddings = np.load(PATH_TO_BERT_WORD_EMBEDDINGS_FILE, allow_pickle=True)

    with TemporaryDirectory() as temp_output_dir:
        alternation_df = pd.read_csv(path_to_input_csv, index_col='verb')
        for frame in alternation_df.columns:
            frame_df = alternation_df[[frame]]
            frame_df = frame_df[frame_df[frame] != 'x'].astype(int)
            run_experiment_for_alternation_df(
                df_labels = frame_df,
                frame = frame,
                output_directory = Path(temp_output_dir),
                word_embeddings = word_embeddings
            )

        for frame in alternation_df.columns:
            computed_evaluation_metrics_file = (
                Path(temp_output_dir) / "static" / f"{frame}_evaluation_metrics.csv"
            )
            # just check that the file is there, since its contents have been tested above.
            assert computed_evaluation_metrics_file.exists()

            computed_output_predictions_file = (
                Path(temp_output_dir) / "static" / f"{frame}_predictions.csv"
            )
            assert computed_output_predictions_file.exists()

            # computed_output_predictions_df = pd.read_csv(computed_output_predictions_file, index_col='verb').sort_values(by='verb')
            # expected_frame_predictions_df = expected_predictions_df[['fold', f'{frame}_true', f'{frame}_predicted']].sort_values(by='verb')
            # assert_frame_equal(expected_frame_predictions_df, computed_output_predictions_df)

def test_run_experiment_for_alternation_df_context():
    """Test the run_experiment_for_alternation_csv() function for contextual embeddings."""

    path_to_input_csv = DATA_DIR / "labelled_data.csv"
    layer_embeddings = np.load(PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE, allow_pickle=True)

    with TemporaryDirectory() as temp_output_dir:
        for i in range(layer_embeddings.shape[1]):
            word_embeddings = layer_embeddings[:, i, :]
            alternation_df = pd.read_csv(path_to_input_csv, index_col='verb')

            for frame in alternation_df.columns:
                frame_df = alternation_df[[frame]]
                frame_df = frame_df[frame_df[frame] != 'x'].astype(int)
                layer = i + 1
                run_experiment_for_alternation_df(
                    df_labels = frame_df,
                    frame = frame,
                    output_directory = Path(temp_output_dir),
                    word_embeddings = word_embeddings,
                    layer = layer
                )

            for frame in alternation_df.columns:
                computed_evaluation_metrics_file = (
                    Path(temp_output_dir) / str(layer) / f"{frame}_evaluation_metrics.csv"
                )
                # just check that the file is there, since its contents have been tested above.
                assert computed_evaluation_metrics_file.exists()

                computed_output_predictions_file = (
                    Path(temp_output_dir) / str(layer) / f"{frame}_predictions.csv"
                )
                assert computed_output_predictions_file.exists()
