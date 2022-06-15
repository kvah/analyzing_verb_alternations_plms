"""
Conduct an experiment whereby we see if we can predict the legitimate
syntactic frames of verbs based on their static or contextual word embeddings.

Authors
-------
James V. Bruno (jbruno@uw.edu)
David Yi (davidyi6@uw.edu)
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
import torch
from typing import List

from alternationprober.constants import (
    PATH_TO_BERT_WORD_EMBEDDINGS_FILE,
    PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE,
    PATH_TO_LAVA_DIR,
    PATH_TO_LAVA_VOCAB,
    PATH_TO_RESULTS_DIRECTORY,
)


# Load vocabulary file mapping verbs to their index in the embedding array.
with PATH_TO_LAVA_VOCAB.open("r") as f:
    VOCAB_TO_INDEX = json.load(f)

# And create the handy reverse lookup, too.
INDEX_TO_VOCAB = {index: vocab for vocab, index in VOCAB_TO_INDEX.items()}

# Set some constants used in cross-validation.
RANDOM_STATE = 575
NUMBER_XVAL_FOLDS = 4


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


def run_experiment_for_alternation_df(
    df_labels: pd.DataFrame, frame:str, output_directory: Path, word_embeddings: torch.Tensor, layer: int = None
) -> None:
    """Train and evaluate a LogisticRegression classifer on the data in ``alternation_df``.
    Two result files will be written to ``output_directory``:
        * a file of predictions: ``*_predictions.csv``
        * a file of evaluation metrics: ``*_evaluation_metrics.csv``
    Parameters
    ----------
    df_labels : pd.DataFrame
        pandas DataFrame with gold standard binary labels for a specific syntactic frame
    frame: 
        which verb frame to use for this experiment, e.g. `causative`
    output_directory : Path
        path to output directory.
    word_embeddings: torch.Tensor
        word embedding tensor of shape (instances, classes)
    """

    # Get the indices for the verbs in this dataset.
    indices_for_X = [VOCAB_TO_INDEX[verb] for verb in df_labels.index]

    # Prepare the data structure for the gold standard labels.
    # This is an array of shape (|instances|, |classes|).
    Y = df_labels[frame]

    # And now get the embeddings to use as features.
    # This is an array of shape (|instances|, 768).
    X = word_embeddings[indices_for_X]

    # Implement 4-fold cross-validation, following the paper.
    cross_validation_iterator = StratifiedKFold(
        n_splits=NUMBER_XVAL_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # Create a list to hold dictionaries of classification results for each instance.
    results = []

    # Iterate over our 4 cross-validation folds and train a classifier for each fold.
    for fold, (train_indices, test_indices) in enumerate(
        cross_validation_iterator.split(X, Y), start=1
    ):

        X_test = X[test_indices]
        Y_test = Y[test_indices]
        verbs_test = [INDEX_TO_VOCAB[index] for index in test_indices]

        X_train = X[train_indices]
        Y_train = Y[train_indices]

        # Define a classifier.
        # The performance was abysmal when we used the L2 penalty, even with CV.
        # I guess we're not worried about overfitting for this.
        if len(np.unique(Y_train)) < 2:
            # sklearn LR will error if there's only one class
            classifier = DummyClassifier(strategy="most_frequent")
        else:
            classifier = LogisticRegression(penalty="none")
        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_test)

        # Iterate over instances and collect true vs. predictions for each alternation.
        for i in range(len(Y_test)):
            
            verb = Y_test.index[i]
            true_label = Y_test[i]
            prediction = predictions[i]

            print(verb, true_label, prediction)

            # Start to build up a dictinoary for this instance.
            result_dict = {"verb": verb, "fold": fold}

            # Add classification results and true labels to the dictionary.
            result_dict[f"{frame}_true"] = true_label
            result_dict[f"{frame}_predicted"] = prediction
            results.append(result_dict)

    # Collect the results together into a nice DataFrame to output.
    predictions_df = pd.DataFrame.from_records(results)

    # Store the predictions for future reference.
    if layer:
        layer_dir = output_directory / str(layer)
        layer_dir.mkdir(parents=True, exist_ok=True)
        prediction_file = layer_dir / f"{frame}_predictions.csv"
    else:
        static_dir = output_directory / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        prediction_file = static_dir / f"{frame}_predictions.csv"
    predictions_df.to_csv(prediction_file, index=False)

    # Get evaluation metrics.
    evaluation_df = get_evaluation_df(predictions_df, df_labels.columns.tolist())
    print(evaluation_df)

    # And store them.
    if layer:
        evaluation_file = layer_dir / f"{frame}_evaluation_metrics.csv"
        evaluation_df.to_csv(evaluation_file, index=False)
    else:
        evaluation_file = static_dir / f"{frame}_evaluation_metrics.csv"
    evaluation_df.to_csv(evaluation_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run an experiment to try to predict the syntactic "
        "frames of verbs based on their static/contextual word embeddings."
    )
    parser.add_argument(
        "--output_directory",
        help="output_directory for experimental_results",
        default=(PATH_TO_RESULTS_DIRECTORY / "linear-probe-for-word-embeddings"),
        type=Path,
        nargs="?",
    )
    parser.add_argument(
        "--use_context_embeddings",
        help="whether to use contextual bert embeddings instead of static wordpiece embeddings",
        action="store_true"
    )
    args = parser.parse_args()

    alternation_csv = PATH_TO_LAVA_DIR / 'verb_frames.csv'
    alternation_df = pd.read_csv(alternation_csv, index_col='verb')

    args.output_directory.mkdir(parents=True, exist_ok=True)

    if args.use_context_embeddings:
        try:
            # Load context word embeddings.
            layer_embeddings = np.load(PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE, allow_pickle=True)
        except FileNotFoundError as e:
            message = f"""
            {PATH_TO_BERT_CONTEXT_WORD_EMBEDDINGS_FILE} not found.  
            Execute 'get_bert_context_word_embeddings` before continuing.
            """
            raise FileNotFoundError(message) from e
    
    else:
        try:
            # Load static word embeddings.
            word_embeddings = np.load(PATH_TO_BERT_WORD_EMBEDDINGS_FILE, allow_pickle=True)
        except FileNotFoundError as e:
            message = f"""
            {PATH_TO_BERT_WORD_EMBEDDINGS_FILE} not found.  
            Execute 'get_bert_word_embeddings` before continuing.
            """
            raise FileNotFoundError(message) from e


    if args.use_context_embeddings:
        # Loop over each context layer
        for i in range(layer_embeddings.shape[1]):
            word_embeddings = layer_embeddings[:, i, :]
            print(f'Layer {i+1}')
            for frame in alternation_df.columns:
                frame_df = alternation_df[[frame]]
                # Remove verbs with missing values
                frame_df = frame_df[frame_df[frame] != 'x'].astype(int)
                print('----------------------------------------')
                print(f"running experiment on {frame}")
                run_experiment_for_alternation_df(
                    frame_df, frame, args.output_directory, word_embeddings, layer=i+1
                )
            print('\n')
    else:
        for frame in alternation_df.columns:
            frame_df = alternation_df[[frame]]
            # Remove verbs with missing values
            frame_df = frame_df[frame_df[frame] != 'x'].astype(int)
            print(f"running experiment on {frame}")
            run_experiment_for_alternation_df(frame_df, frame, args.output_directory, word_embeddings)


if __name__ == "__main__":
    main()
