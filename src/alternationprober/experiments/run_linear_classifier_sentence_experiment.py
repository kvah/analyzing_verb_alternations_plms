"""
Conduct an experiment whereby we see if we can predict a sentence with given verb grammatical
or ungrammatical. Sentence embeddings are based on hidden layers in BERT you designate.

Authors
-------
Jiayu Han (jyhan126@uw.edu)
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from alternationprober.embeddings.get_bert_sentence_embeddings import get_sent_embeddings_dataset
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from typing import List

from alternationprober.constants import (
    PATH_TO_FAVA_DIR,
    PATH_TO_RESULTS_DIRECTORY,
)


def run_experiment_for_an_alternation(alternation_category: Path, layers: List, outputdir: Path,
                                      writetofile: bool) -> None:
    '''
    alternation_category: e.g. ./ling-575-analyzing-nn-group/results/linear-probe-for-sentence-embeddings/understood
    layers: [12] default is the last layer, you can set any combination of layers
    outputdir: ./ling-575-analyzing-nn-group/results/linear-probe-for-sentence-embeddings
    writetofile: default is true. In default, it will save the full predicted values with original dataset to help future
                analysis, if you want to reproduce one specific experiment to quickly see the confusion matrix, you can set it as false

    '''
    X, y = get_sent_embeddings_dataset(alternation_category / 'train.tsv', layers)
    X_dev, y_dev = get_sent_embeddings_dataset(alternation_category / 'dev.tsv', layers)
    X_test, y_test = get_sent_embeddings_dataset(alternation_category / 'test.tsv', layers)

    train = pd.read_csv(alternation_category / 'train.tsv', sep='\t+', names=['alternation', 'label', 'sentence'])
    dev = pd.read_csv(alternation_category / 'dev.tsv', sep='\t+', names=['alternation', 'label', 'sentence'])
    test = pd.read_csv(alternation_category / 'test.tsv', sep='\t+', names=['alternation', 'label', 'sentence'])
    # add a new column to store the predicted values
    new_add = pd.DataFrame(columns=['pred_label'])

    #followings will have [alternation, label, sentence, predicted label]
    train = pd.concat([train, new_add])
    dev = pd.concat([dev, new_add])
    test = pd.concat([test, new_add])


    #set outputname by the layers used in probing, e.g. 12dative.tsv, 1_12dative.tsv
    outputname = "_".join([str(a) for a in layers])

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
    filedir = outputdir / alternation_category.parts[-1]
    filedir.mkdir(parents=True, exist_ok=True)

    #if you do not need these readable results, set writetofile as false in args
    if writetofile:
        train.to_csv(outputdir / alternation_category.parts[-1] / (outputname + 'train.tsv'), sep='\t', index=False,
                     header=True)
        test.to_csv(outputdir / alternation_category.parts[-1] / (outputname + 'test.tsv'), sep='\t', index=False,
                    header=True)
        train.to_csv(outputdir / alternation_category.parts[-1] / (outputname + 'dev.tsv'), sep='\t', index=False,
                     header=True)

    mcc_train = matthews_corrcoef(y, y_pred)
    mcc_dev = matthews_corrcoef(y_dev, y_dev_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)

    acc_train = accuracy_score(y, y_pred)
    acc_dev = accuracy_score(y_dev, y_dev_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    print(
        f'{alternation_category.parts[-1]}:mcc of training dataset:{mcc_train}, mcc of dev dataset :{mcc_dev}, mcc of testing dataset:{mcc_test}')
    print(
        f'{alternation_category.parts[-1]}:acc of training dataset:{acc_train}, acc of dev dataset :{acc_dev}, acc of testing dataset:{acc_test}')

    # to get a direct sense of confusion matrix
    mtrain = pd.DataFrame(confusion_matrix(y, y_pred, labels=[1, 0]),
                          index=['true:0', 'true:1'],
                          columns=['pred:0', 'pred:1'])
    mdev = pd.DataFrame(confusion_matrix(y_dev, y_dev_pred, labels=[1, 0]),
                        index=['true:0', 'true:1'],
                        columns=['pred:0', 'pred:1'])
    mtest = pd.DataFrame(confusion_matrix(y_test, y_test_pred, labels=[1, 0]),
                         index=['true:0', 'true:1'],
                         columns=['pred:0', 'pred:1'])

    print(f'train:confusion matrix for {alternation_category}\n', mtrain)
    print(f'dev:confusion matrix for {alternation_category}\n', mdev)
    print(f'test:confusion matrix for {alternation_category}\n', mtest)

    #save mcc and acc for logistic regression
    result = {
        "alternation": alternation_category.parts[-1],
        "layers": layers,
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
    result = run_experiment_for_an_alternation(alternation_folder_path, args.probing_layers, args.output_directory,
                                               args.notwritetofile)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an experiment to try to predict the acceptability of"
                    " a sentence with given verb the syntactic "
    )
    parser.add_argument(
        "--output_directory",
        help="output_directory for exerimental_results",
        default=(PATH_TO_RESULTS_DIRECTORY / "linear-probe-for-sentence-embeddings"),
        type=Path,
        nargs="?",
    )
    parser.add_argument(
        "--alternation",
        help="alternation category ['combined', 'dative', 'inchoative', 'spray_load', 'there', 'understood']",
        default='understood',
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--probing_layers",
        help="layers in Bert: [12](last layer), [0,1,2,3,4,5,6,7,8,9,10,11,12](all layers)",
        default=[1],
        type=List,
        nargs="?",
    )
    parser.add_argument(
        "--resultsfilename",
        help="the file storing a series of probing results",
        default='sentence_probing.csv',
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--notwritetofile",
        help="whether writing to files",
        action="store_false",
    )

    # in the args, you can use --notwritetofile to control whether you need to save files for future analysis, default will save
    # you can use --probing_layers to designate which layers you want to probe,
    # also, you can use --alternation to choose which class you want to probe.
    args = parser.parse_args()

    # main part, result is a dict, then the new result will be added in the all_results.json file containing all previous results

    # result: {'alternation': 'understood', 'layers': [12], 'training_accuracy': 1.0, 'dev_accuracy': 0.8777777777777778,
    # 'testing_accuracy': 0.9088050314465409, 'training_mcc': 1.0, 'dev_mcc': 0.7648661601319408, 'test_mcc': 0.8164495718788796,
    # 'training_accuracy_bl': 0.5641891891891891, 'dev_accuracy_bl': 0.5, 'test_accuracy_bl': 0.5377358490566038}

    result = main(args)
    with open(args.output_directory / "all_results.json", 'a') as outfile:
        json.dump(result, outfile)
        outfile.write('\n')
        outfile.close()

    # following codes are using a for loop to iterate on every alternation class with same setting(whether store, same layers)
    # for alternation in ['combined', 'dative', 'inchoative', 'spray_load', 'there', 'understood']:
    #     args.alternation = alternation
    #     result = main(args)
    #     with open(args.output_directory/"all_results.json", 'a') as outfile:
    #         json.dump(result, outfile)
    #         outfile.write('\n')
    #         outfile.close()
