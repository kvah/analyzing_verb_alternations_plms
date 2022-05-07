"""
Tests for the experiments.run_linear_classifier_experiment module
"""
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
from tempfile import TemporaryDirectory


from alternationprober.experiments.run_linear_classifier_experiment import (
    get_evaluation_df,
    run_experiment_for_alternation_csv,
)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"


def test_get_evaluation_df():
    """Test the get_evaluation_df() function."""
    path_to_expected_output_df = DATA_DIR / "labelled_data_evaluation_metrics.csv"
    expected_output_df = pd.read_csv(path_to_expected_output_df)

    path_to_predictions_df = DATA_DIR / "labelled_data_predictions.csv"
    predictions_df = pd.read_csv(path_to_predictions_df)

    alternation_classes = ["subclass1", "subclass2", "subclass3"]

    computed_output_df = get_evaluation_df(predictions_df, alternation_classes)

    assert_frame_equal(expected_output_df, computed_output_df)


def test_run_experiment_for_alternation_csv():
    """Test the run_experiment_for_alternation_csv() function."""
    path_to_input_csv = DATA_DIR / "labelled_data.csv"

    path_to_expected_predictions_csv = DATA_DIR / "labelled_data_predictions.csv"
    expected_predictions_df = pd.read_csv(path_to_expected_predictions_csv)

    with TemporaryDirectory() as temp_output_dir:
        run_experiment_for_alternation_csv(path_to_input_csv, Path(temp_output_dir))

        computed_evaluation_metrics_file = (
            Path(temp_output_dir) / "labelled_data_evaluation_metrics.csv"
        )

        # just check that the file is there, since its contents have been tested above.
        assert computed_evaluation_metrics_file.exists()

        computed_output_predictions_file = (
            Path(temp_output_dir) / "labelled_data_predictions.csv"
        )
        computed_output_predictions_df = pd.read_csv(computed_output_predictions_file)

    assert_frame_equal(expected_predictions_df, computed_output_predictions_df)
