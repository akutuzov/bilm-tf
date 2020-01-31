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
import torch
from torch.utils.data import DataLoader
import transformers as ppb # pytorch transformers
warnings.filterwarnings("ignore")


def classify(data_file, w2v=None, elmo=None, bert=None, max_batch_size=300, algo='logreg'):
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

    elif bert:
        tokenizer, model = bert
        tokenized0 = data.text0.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        tokenized1 = data.text1.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        print('Padding...', file=sys.stderr)
        max_len = 0
        for i in tokenized0.values + tokenized1.values:
            if len(i) > max_len:
                max_len = len(i)
        print('Max length:', max_len)

        padded0 = [i + [0]*(max_len-len(i)) for i in tokenized0.values]
        padded1 = [i + [0]*(max_len-len(i)) for i in tokenized1.values]

        input_ids0 = torch.tensor(np.array(padded0)).to('cuda')
        input_ids1 = torch.tensor(np.array(padded1)).to('cuda')

        features = []
        for inp in [input_ids0, input_ids1]:
            loader = DataLoader(inp, batch_size=256, shuffle=False)
            last_hidden_states = []
            with torch.no_grad():
                for i in loader:
                    last_hidden_states.append(model(i))
            last_hidden_states = torch.cat([i[0] for i in last_hidden_states], 0)
            print('BERT output shape:', last_hidden_states.shape, file=sys.stderr)

            # Slice the output for the first position for all the sequences, take all hidden unit outputs
            # features.append(last_hidden_states[:,0,:].cpu().numpy())

            # Take the average embedding for all the sequences:
            features.append([np.mean(row, axis=0) for row in last_hidden_states.cpu().numpy()])

        train0 = features[0]
        train1 = features[1]

    classes = Counter(y)
    print('Distribution of classes in the whole sample:', dict(classes))

    x_train = [[np.dot(t0, t1)] for t0, t1 in zip(train0, train1)]
    print('Train shape:', len(x_train))

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
    arg('--bert', help='Path to BERT model (optional)')
    arg('--elmo', help='Path to ELMo model (optional)')
    parser.set_defaults(w2v=False)
    parser.set_defaults(elmo=False)

    args = parser.parse_args()
    data_path = args.input

    if args.bert:
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, args.bert)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights).to('cuda')
        eval_scores = classify(data_path, bert=(tokenizer, model))
    elif args.elmo:
        emb_model = load_elmo_embeddings(args.elmo, top=False)
        eval_scores = classify(data_path, elmo=emb_model)
    else:
        eval_scores = classify(data_path)
