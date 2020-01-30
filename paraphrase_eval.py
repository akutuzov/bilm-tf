# python3
# coding: utf-8

import argparse
import warnings
from collections import Counter
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from wsd_helpers import *
import random
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")


def classify(data_file, w2v=None, elmo=None, max_batch_size=300, algo='logreg'):
    data = pd.read_csv(data_file, sep='\t', compression='gzip')
    print(data.head())

    train0 = []
    train1 = []
    y = data.label.values
    if elmo:
        batcher, sentence_character_ids, elmo_sentence_input = elmo
        sentences0 = [t.split() for t in data.text0]
        sentences1 = [t.split() for t in data.text1]
        print('=====')
        print('%d sentences total' % (len(sentences0)))
        print('=====')
        # Here we divide all the sentences for the current word in several chunks
        # to reduce the batch size
        with tf.Session() as sess:
            # It is necessary to initialize variables once before running inference.
            sess.run(tf.global_variables_initializer())
            for chunk in divide_chunks(sentences0, max_batch_size):
                train0 += get_elmo_vector_average(sess, chunk, batcher, sentence_character_ids, elmo_sentence_input)
            for chunk in divide_chunks(sentences1, max_batch_size):
                train1 += get_elmo_vector_average(sess, chunk, batcher, sentence_character_ids, elmo_sentence_input)

    classes = Counter(y)
    print('Distribution of classes in the whole sample:', dict(classes))

    x_train = [[np.dot(t0, t1)] for t0, t1 in zip(train0, train1)]
    # print('Train entry shape:', x_train[0].shape)

    if algo == 'logreg':
        clf = LogisticRegression(solver='lbfgs', max_iter=2000, multi_class='auto', class_weight='balanced')
    else:
        clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=500)
    dummy = DummyClassifier(strategy='stratified')
    averaging = True  # Do you want to average the cross-validate metrics?

    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    # some splits are containing samples of one class, so we split until the split is OK
    counter = 0
    while True:
        try:
            cv_scores = cross_validate(clf, x_train, y, cv=10, scoring=scoring)
            cv_scores_dummy = cross_validate(dummy, x_train, y, cv=10, scoring=scoring)
        except ValueError:
            counter += 1
            if counter > 500:
                print('Impossible to find a good split!')
                exit()
            continue
        else:
            # No error; stop the loop
            break

    scores = ([cv_scores['test_precision_macro'].mean(), cv_scores['test_recall_macro'].mean(), cv_scores['test_f1_macro'].mean()])
    dummy_scores = ([cv_scores_dummy['test_precision_macro'].mean(), cv_scores_dummy['test_recall_macro'].mean(), cv_scores_dummy['test_f1_macro'].mean()])
    print('Real scores:')
    print('=====')
    print('Precision: %0.3f' % scores[0])
    print('Recall: %0.3f' % scores[1])
    print('F1: %0.3f' % scores[2])

    print('Random choice scores:')
    print('=====')
    print('Precision: %0.3f' % dummy_scores[0])
    print('Recall: %0.3f' % dummy_scores[1])
    print('F1: %0.3f' % dummy_scores[2])


    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', help='Path to tab-separated file with paraphrase data', required=True)
    arg('--w2v', help='Path to word2vec model (optional)')
    arg('--elmo', help='Path to ELMo model (optional)')
    parser.set_defaults(w2v=False)
    parser.set_defaults(elmo=False)

    args = parser.parse_args()
    data_path = args.input

    if args.w2v:
        emb_model = load_word2vec_embeddings(args.w2v)
        eval_scores = classify(data_path, w2v=emb_model)
    elif args.elmo:
        emb_model = load_elmo_embeddings(args.elmo, top=False)
        eval_scores = classify(data_path, elmo=emb_model)
    else:
        eval_scores = classify(data_path)
