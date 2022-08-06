"""
Conduct an experiment whereby we see if we can predict a sentence with given verb grammatical
or ungrammatical. Sentence embeddings are based on hidden layers in BERT you designate.

Authors
-------
Jiayu Han (jyhan126@uw.edu)
David Yi (davidyi6@uw.edu)
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from alternationprober.embeddings.get_bert_sentence_embeddings import get_sent_embeddings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from typing import List

from alternationprober.constants import (
    PATH_TO_FAVA_DIR,
    PATH_TO_RESULTS_DIRECTORY,
    PATH_TO_SENTENCE_EMBEDDINGS_DIR
)


def run_experiment_for_an_alternation(
    alternation_category: Path, 
    output_dir: Path,
    model_name: str):

    '''
    alternation_category: e.g. ./ling-575-analyzing-nn-group/results/linear-probe-for-sentence-embeddings/understood
    outputdir: ./ling-575-analyzing-nn-group/results/linear-probe-for-sentence-embeddings
    '''

    train = pd.read_csv(alternation_category / 'train.tsv', sep='\t+', names=['alternation', 'label', 'sentence'])
    dev = pd.read_csv(alternation_category / 'dev.tsv', sep='\t+', names=['alternation', 'label', 'sentence'])
    test = pd.read_csv(alternation_category / 'test.tsv', sep='\t+', names=['alternation', 'label', 'sentence'])

    if '/' in args.model_name:
        model_name = args.model_name.split('/')[1]
    else:
        model_name = args.model_name

    train_embeds = np.load(PATH_TO_SENTENCE_EMBEDDINGS_DIR / model_name / args.alternation / 'train.npy')
    dev_embeds = np.load(PATH_TO_SENTENCE_EMBEDDINGS_DIR / model_name / args.alternation / 'dev.npy')
    test_embeds = np.load(PATH_TO_SENTENCE_EMBEDDINGS_DIR / model_name / args.alternation/ 'test.npy')

    train_labels = train['label']
    dev_labels = dev['label']
    test_labels = test['label']

    for layer in range(12):
        X, y = train_embeds[layer], train_labels 
        X_dev, y_dev = dev_embeds[layer], dev_labels
        X_test, y_test = test_embeds[layer], test_labels

        # add a new column to store the predicted values
        new_add = pd.DataFrame(columns=['pred_label'])

        #followings will have [alternation, label, sentence, predicted label]
        train = pd.concat([train, new_add])
        dev = pd.concat([dev, new_add])
        test = pd.concat([test, new_add])

        #here can set logistic regression classifier, here only use L1 regularizaton (compared with L1=0.5, L2=0.5, L1=1 works better).
        classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=1).fit(X, y)

        y_pred = classifier.predict(X)
        y_dev_pred = classifier.predict(X_dev)
        y_test_pred = classifier.predict(X_test)

        train['pred_label'] = y_pred
        dev['pred_label'] = y_dev_pred
        test['pred_label'] = y_test_pred

        # save readable results(alternation, label, sentence, pred_label) to
        # ../../results/linear-probe-for-sentence-embeddings/dative/12.tsv
        train_dir = output_dir / model_name / alternation_category.parts[-1] / 'train'
        dev_dir = output_dir / model_name / alternation_category.parts[-1] / 'dev'
        test_dir = output_dir / model_name / alternation_category.parts[-1] / 'test'

        train_dir.mkdir(parents=True, exist_ok=True)
        dev_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(train_dir / str(layer+1), sep='\t', index=False, header=True)
        dev.to_csv(dev_dir /  str(layer+1), sep='\t', index=False, header=True)
        test.to_csv(test_dir / str(layer+1), sep='\t', index=False, header=True)

        mcc_train = matthews_corrcoef(y, y_pred)
        mcc_dev = matthews_corrcoef(y_dev, y_dev_pred)
        mcc_test = matthews_corrcoef(y_test, y_test_pred)

        acc_train = accuracy_score(y, y_pred)
        acc_dev = accuracy_score(y_dev, y_dev_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        print(f'Results for {model_name}, {args.alternation} layer: {layer}')
        print(
            f"""
            {alternation_category.parts[-1]}: 
            mcc of training dataset: {mcc_train}, 
            mcc of dev dataset : {mcc_dev}, 
            mcc of testing dataset: {mcc_test}
            """)
        print(
            f"""
            acc of training dataset: {acc_train}, 
            acc of dev dataset : {acc_dev}, 
            acc of testing dataset: {acc_test}
            """)

        # # to get a direct sense of confusion matrix
        # mtrain = pd.DataFrame(confusion_matrix(y, y_pred, labels=[1, 0]),
        #                     index=['true:0', 'true:1'],
        #                     columns=['pred:0', 'pred:1'])
        # mdev = pd.DataFrame(confusion_matrix(y_dev, y_dev_pred, labels=[1, 0]),
        #                     index=['true:0', 'true:1'],
        #                     columns=['pred:0', 'pred:1'])
        # mtest = pd.DataFrame(confusion_matrix(y_test, y_test_pred, labels=[1, 0]),
        #                     index=['true:0', 'true:1'],
        #                     columns=['pred:0', 'pred:1'])

        # print(f'train: confusion matrix for {alternation_category}\n', mtrain)
        # print(f'dev: confusion matrix for {alternation_category}\n', mdev)
        # print(f'test: confusion matrix for {alternation_category}\n', mtest)

        #save mcc and acc for logistic regression
        result = {
            "alternation": alternation_category.parts[-1],
            "training_accuracy": acc_train,
            "dev_accuracy": acc_dev,
            "testing_accuracy": acc_test,
            "training_mcc": mcc_train,
            "dev_mcc": mcc_dev,
            "test_mcc": mcc_test
        }

        # majority baseline
        majority_classifier = DummyClassifier(strategy="most_frequent")
        majority_classifier.fit(y, y)
        acc_train_bl = majority_classifier.score(X, y)
        acc_dev_bl = majority_classifier.score(X_dev, y_dev)
        acc_test_bl = majority_classifier.score(X_test, y_test)
        result['training_accuracy_bl'] = acc_train_bl
        result['dev_accuracy_bl'] = acc_dev_bl
        result['test_accuracy_bl'] = acc_test_bl

    return result


def main(args):
    alternation_folder_path = PATH_TO_FAVA_DIR / args.alternation
    result = run_experiment_for_an_alternation(
        alternation_folder_path, 
        args.output_directory,
        args.model_name
    )
    print(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an experiment to try to predict the acceptability of"
                    " a sentence with given verb the syntactic "
    )
    parser.add_argument(
        "--model_name", 
        type=str,
        choices=['bert-base-uncased', 'roberta-base', 'google/electra-base-discriminator', 'microsoft/deberta-base'], 
        default='bert-base-uncased'
    )
    parser.add_argument(
        "--output_directory",
        help="output_directory for exerimental_results",
        default=(PATH_TO_RESULTS_DIRECTORY / "linear-probe-for-sentence-embeddings"),
        type=Path,
        nargs="?",
    )
    parser.add_argument(
        "--resultsfilename",
        help="the file storing a series of probing results",
        default='sentence_probing.csv',
        type=str,
        nargs="?",
    )

    args = parser.parse_args()

    if '/' in args.model_name:
        model_name = args.model_name.split('/')[1]
    else:
        model_name = args.model_name

    for alternation in ['combined', 'dative', 'inchoative', 'spray_load', 'there', 'understood']:
        args.alternation = alternation
        result = main(args)
        with open(args.output_directory/ model_name / "all_results.json", 'a') as outfile:
            json.dump(result, outfile)
            outfile.write('\n')
