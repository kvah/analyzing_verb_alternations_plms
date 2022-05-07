"""
Conduct an experiment whereby we see if we can predict the legitimate
syntactic frames of verbs based on their static word embeddings.

Authors
-------
James V. Bruno (jbruno@uw.edu)
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from typing import List

from alternationprober.constants import (
    PATH_TO_BERT_WORD_EMBEDDINGS_FILE,
    PATH_TO_LAVA_DIR,
    PATH_TO_LAVA_VOCAB,
    PATH_TO_RESULTS_DIRECTORY,
)


# Load vocabulary file mapping verbs to their index in the embedding array.
with PATH_TO_LAVA_VOCAB.open("r") as f:
    VOCAB_TO_INDEX = json.load(f)

# And create the handy reverse lookup, too.
INDEX_TO_VOCAB = {index: vocab for vocab, index in VOCAB_TO_INDEX.items()}

# Load word embeddings.
WORD_EMBEDDINGS = np.load(PATH_TO_BERT_WORD_EMBEDDINGS_FILE, allow_pickle=True)

# Set some constants used in cross-validation.
RANDOM_STATE = 575
NUMBER_XVAL_FOLDS = 4


# This is temporary:
OUTPUT_DIR = Path(".").resolve() / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_evaluation_df(
    predictions_df: pd.DataFrame, alternation_classes: List[str]
) -> pd.DataFrame:
    """Return a dataframe of accuracy and MCC metrics.

    Calculate accuracy and MCC metrics based on the predictions in
    ``predictions_df``.  Also calucate accuracy based on a majority baseline.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        input DataFrame.  Columns must be named according to:
            ``<alternation_class>_true`` or
            ``<alternation_class>_predicted``
        where ``<alternation_class>`` is a member ``alternation_classes``.

    alternation_classes : List[str]
        Input list of alternation classes present in ``predictions_df``.

    Returns
    -------
    evaluation_df : pd.DataFrame
        DataFrame of evaluation metrics.
    """
    output_records = []

    for alternation_class in alternation_classes:
        Y_true = predictions_df[f"{alternation_class}_true"]
        Y_pred = predictions_df[f"{alternation_class}_predicted"]

        # Calculate real evaluation metrics.
        accuracy = accuracy_score(Y_true, Y_pred)
        mcc = matthews_corrcoef(Y_true, Y_pred)

        # Compare against a majority baseline.
        majority_classifier = DummyClassifier(strategy="most_frequent")
        # The first param is ignored in the .fit() method.
        majority_classifier.fit(Y_true, Y_true)
        # And the features don't matter either below.
        majority_predictions = majority_classifier.predict(Y_true)

        majority_accuracy = accuracy_score(Y_true, majority_predictions)
        # The MCC is always zero for the majority baseline.  Don't bother.
        # majority_mcc = matthews_corrcoef(Y_true, majority_predictions)

        output_records.append(
            {
                "alternation_class": alternation_class,
                "accuracy": accuracy,
                "baseline_accuracy": majority_accuracy,
                "mcc": mcc,
            }
        )

    evaluation_df = pd.DataFrame.from_records(output_records)

    return evaluation_df


def run_experiment_for_alternation_csv(
    alternation_csv: Path, output_directory: Path
) -> None:
    """Train and evaluate a LogisticRegression classifer on the data in ``alternation_csv``.

    Two result files will be written to ``output_directory``:
        * a file of predictions: ``*_predictions.csv``
        * a file of evaluation metrics: ``*_evaluation_metrics.csv``

    Parameters
    ----------
    alternation_csv : Path
        path to alternation csv with gold standard binary labels.
    output_directory : Path
        path to output directory.
    """
    # Load the input data.
    df_labels = pd.read_csv(alternation_csv)

    # Get the indices for the verbs in this dataset.
    indices_for_X = [VOCAB_TO_INDEX[verb] for verb in df_labels["verb"]]

    # Set the verb as the index of the dataframe to make the numpy conversion clean.
    df_labels = df_labels.set_index("verb")

    # Prepare the data structure for the gold standard labels.
    # This is an array of shape (|instances|, |classes|).
    Y = df_labels.to_numpy()

    # And now get the embeddings to use as features.
    # This is an array of shape (|instances|, 768).
    X = WORD_EMBEDDINGS[indices_for_X]

    # Implement 4-fold cross-validation, following the paper.
    cross_validation_iterator = StratifiedKFold(
        n_splits=NUMBER_XVAL_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # It's nice to use stratified cross-validation to make sure we get an
    # even representation of classes across folds, but we can't do that
    # with a multi-label classification problem.  We'll try to hack it to get
    # close by summing the binary labels, so we know there are at least some
    # positive cases in each fold.
    collapsed_Y = Y.sum(axis=1)

    # Create a list to hold dictionaries of classification results for each instance.
    results = []

    # Iterate over our 4 cross-validation folds and train a classifier for each fold.
    for fold, (train_indices, test_indices) in enumerate(
        cross_validation_iterator.split(X, collapsed_Y), start=1
    ):

        X_test = X[test_indices]
        Y_test = Y[test_indices]
        verbs_test = [INDEX_TO_VOCAB[index] for index in test_indices]

        X_train = X[train_indices]
        Y_train = Y[train_indices]

        # Define a classifier.
        classifier = MultiOutputClassifier(
            LogisticRegression(penalty="none")
            # The performance was abysmal when we used the L2 penalty, even with CV.
            # I guess we're not worried about overfitting for this.
        )
        classifier.fit(X_train, Y_train)

        predictions = classifier.predict(X_test)

        # Iterate over instances and collect true vs. predictions for each alternation.
        for verb, true_label, prediction in zip(verbs_test, Y_test, predictions):

            # Start to build up a dictinoary for this instance.
            result_dict = {"verb": verb, "fold": fold}

            # The array_index corresponds exactly to the column index from the input df.
            for array_index, alternation_class in enumerate(df_labels.columns):

                # Add classification results and true labels to the dictionary.
                result_dict[f"{alternation_class}_true"] = true_label[array_index]
                result_dict[f"{alternation_class}_predicted"] = prediction[array_index]

            results.append(result_dict)

    # Collect the results together into a nice DataFrame to output.
    predictions_df = pd.DataFrame.from_records(results)

    # Store the predictions for future reference.
    prediction_file = output_directory / f"{alternation_csv.stem}_predictions.csv"
    predictions_df.to_csv(prediction_file, index=False)

    # Get evaluation metrics.
    evaluation_df = get_evaluation_df(predictions_df, df_labels.columns.tolist())

    print(evaluation_df)

    # And store them.
    evaluation_file = (
        output_directory / f"{alternation_csv.stem}_evaluation_metrics.csv"
    )
    evaluation_df.to_csv(evaluation_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run an experiment to try to predict the syntactic "
        "frames of verbs based on their static word embeddings."
    )
    parser.add_argument(
        "output_directory",
        help="output_directory for exerimental_results",
        default=(PATH_TO_RESULTS_DIRECTORY / "linear-probe-for-word-embeddings"),
        type=Path,
        nargs="?",
    )
    args = parser.parse_args()

    alternation_csvs = [
        csv_file
        for csv_file in PATH_TO_LAVA_DIR.glob("*.csv")
        if not csv_file.stem == "all_verbs"
    ]

    args.output_directory.mkdir(parents=True, exist_ok=True)

    for alternation_csv in alternation_csvs:
        print(f"running experiment on {alternation_csv}")

        run_experiment_for_alternation_csv(alternation_csv, args.output_directory)


if __name__ == "__main__":
    main()
