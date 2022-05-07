"""
Conduct an experiment whereby we see if we can predict the legitimate
syntactic frames of verbs based on their static word embeddings.

Authors
-------
James V. Bruno (jbruno@uw.edu)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier

from alternationprober.constants import (
    PATH_TO_BERT_WORD_EMBEDDINGS_FILE,
    PATH_TO_LAVA_DIR,
    PATH_TO_LAVA_VOCAB,
)


RANDOM_STATE = 575
NUMBER_XVAL_FOLDS = 4


OUTPUT_DIR = Path(".").resolve() / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_evaluation_df(predictions_df, alternation_classes):

    output_records = []

    for alternation_class in alternation_classes:
        Y_true = predictions_df[f"{alternation_class}_true"]
        Y_pred = predictions_df[f"{alternation_class}_predicted"]

        accuracy = accuracy_score(Y_true, Y_pred)
        mcc = matthews_corrcoef(Y_true, Y_pred)

        output_records.append(
            {"alternation_class": alternation_class, "accuracy": accuracy, "mcc": mcc}
        )

    evaluation_df = pd.DataFrame.from_records(output_records)

    return evaluation_df


def main():
    # Load mapping of vocabulary to embedding index.
    with PATH_TO_LAVA_VOCAB.open("r") as f:
        vocab_to_index = json.load(f)

    index_to_vocab = {index: vocab for vocab, index in vocab_to_index.items()}

    # Load word embeddings.
    word_embeddings = np.load(PATH_TO_BERT_WORD_EMBEDDINGS_FILE, allow_pickle=True)

    # Load spray_load data.
    df_labels = pd.read_csv(PATH_TO_LAVA_DIR / "sl.csv")

    # Get the indices for the verbs in this dataset.
    indices_for_X = [vocab_to_index[verb] for verb in df_labels["verb"]]

    # Set the verb as the index of the dataframe to make the numpy conversion clean.
    df_labels = df_labels.set_index("verb")

    # Prepare the data structure for the gold standard labels.
    # This is an array of shape (|instances|, |classes|).
    Y = df_labels.to_numpy()

    # And now get the embeddings to use as features.
    X = word_embeddings[indices_for_X]

    # Implement 4-fold cross-validation following the paper.
    cross_validation_iterator = StratifiedKFold(
        n_splits=NUMBER_XVAL_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # It's nice to use stratified cross-validation to make sure we get an
    # even representation of classes across folds, but we can't do that
    # with a multi-label classification problem.  Try to hack it to get
    # close by summing the binary labels, so we know there are at least some
    # positive cases in each fold.
    collapsed_Y = Y.sum(axis=1)

    # Create a list to hold dictionaries of results for each instance
    results = []

    # Iterate over our 4 cross-validation folds and train a classifier
    for fold, (train_indices, test_indices) in enumerate(
        cross_validation_iterator.split(X, collapsed_Y), start=1
    ):

        test_verbs = [index_to_vocab[index] for index in test_indices]

        X_train = X[train_indices]
        X_test = X[test_indices]

        Y_train = Y[train_indices]
        Y_test = Y[test_indices]

        # Define a classifier.
        classifier = MultiOutputClassifier(
            LogisticRegression(penalty='none')
            # The performance was abysmal when we used the L2 penalty, even with CV.
            # I guess we're not worried about overfitting for this.
        )
        classifier.fit(X_train, Y_train)

        predictions = classifier.predict(X_test)

        # Iterate over instances and collection true vs. predictions for each alternation.
        for verb, true_label, prediction in zip(test_verbs, Y_test, predictions):

            result_dict = {"verb": verb, "fold": fold}

            # the array_index corresponds exactly to the column index from the input df.
            for array_index, alternation_class in enumerate(df_labels.columns):

                result_dict[f"{alternation_class}_true"] = true_label[array_index]
                result_dict[f"{alternation_class}_predicted"] = prediction[array_index]

            results.append(result_dict)

    prediction_file = OUTPUT_DIR / "predictions.csv"
    predictions_df = pd.DataFrame.from_records(results)
    predictions_df.to_csv(prediction_file, index=False)

    evaluation_df = get_evaluation_df(predictions_df, df_labels.columns.tolist())

    print(evaluation_df)

    evaluation_file = OUTPUT_DIR / "evaluation_metrics.csv"
    evaluation_df.to_csv(evaluation_file)


if __name__ == "__main__":
    main()
