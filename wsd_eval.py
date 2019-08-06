# python3
# coding: utf-8

import argparse
import warnings
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from wsd_helpers import *

warnings.filterwarnings("ignore")


def classify(data_file, w2v=None, elmo=None, max_batch_size=100):
    data = load_dataset(data_file)
    scores = []

    # data looks like {w1 = [[w1 context1, w1 context2, ...], [w2 context1, w2 context2, ...]], ...}
    for word in data:
        print(word)
        x_train = []
        y = []
        if elmo:
            batcher, sentence_character_ids, elmo_sentence_input = elmo
            sentences = [tokenize(el[0]) for el in data[word]]
            nums = [el[1] for el in data[word]]
            y = [el[2] for el in data[word]]
            input_data = [(s, n) for s, n in zip(sentences, nums)]
            # Here we divide all the sentences for the current word in several chunks
            # to to reduce the batch size
            for chunk in divide_chunks(input_data, max_batch_size):
                chunk_sentences = [el[0] for el in chunk]
                chunk_nums = [el[1] for el in chunk]
                x_train += get_elmo_vector(chunk_sentences, batcher, sentence_character_ids,
                                           elmo_sentence_input, chunk_nums)
        else:
            for instance in data[word]:
                sent, num, cl = instance
                if w2v:
                    vect = get_word_vector(tokenize(sent), w2v, num)
                else:
                    vect = get_dummy_vector()
                x_train.append(vect)
                y.append(cl)
        classes = Counter(y)
        print('Distribution of classes in the whole sample:', dict(classes))

        clf = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto',
                                 class_weight='balanced')
        averaging = True  # Do you want to average the cross-validate metrics?

        scoring = ['precision_macro', 'recall_macro', 'f1_macro']
        # some splits are containing samples of one class, so we split until the split is OK
        counter = 0
        while True:
            try:
                cv_scores = cross_validate(clf, x_train, y, cv=5, scoring=scoring)
            except ValueError:
                counter += 1
                if counter > 500:
                    print('Impossible to find a good split!')
                    exit()
                continue
            else:
                # No error; stop the loop
                break

        scores.append([cv_scores['test_precision_macro'].mean(),
                       cv_scores['test_recall_macro'].mean(), cv_scores['test_f1_macro'].mean()])
        if averaging:
            print("Average Precision on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_precision_macro'].mean(),
                cv_scores['test_precision_macro'].std() * 2), file=sys.stderr)
            print("Average Recall on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_recall_macro'].mean(),
                cv_scores['test_recall_macro'].std() * 2), file=sys.stderr)
            print("Average F1 on 5-fold cross-validation: %0.3f (+/- %0.3f)" % (
                cv_scores['test_f1_macro'].mean(),
                cv_scores['test_f1_macro'].std() * 2), file=sys.stderr)
        else:
            print("Precision values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_precision_macro'], file=sys.stderr)
            print("Recall values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_recall_macro'], file=sys.stderr)
            print("F1 values on 5-fold cross-validation:", file=sys.stderr)
            print(cv_scores['test_f1_macro'], file=sys.stderr)

        print('\n')

    print('Average precision value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[0] for x in scores])), np.std([x[0] for x in scores]) * 2))
    print('Average recall value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[1] for x in scores])), np.std([x[1] for x in scores]) * 2))
    print('Average F1 value for all words: %0.3f (+/- %0.3f)' %
          (float(np.mean([x[2] for x in scores])), np.std([x[2] for x in scores]) * 2))
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', help='Path to tab-separated file with WSD data', required=True)
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
        emb_model = load_elmo_embeddings(args.elmo)
        eval_scores = classify(data_path, elmo=emb_model)
    else:
        eval_scores = classify(data_path)
