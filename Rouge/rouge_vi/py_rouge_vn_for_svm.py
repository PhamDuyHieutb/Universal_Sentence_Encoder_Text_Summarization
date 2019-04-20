from __future__ import division
import collections
import os
import six
import time
from pyvi import ViTokenizer
from definitions import ROOT_DIR


def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)


def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))


def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)


def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result


def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def _safe_f1(matches, recall_total, precision_total, alpha):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return precision_score, recall_score, (precision_score * recall_score) / denom
    else:
        return 0.0, 0.0, 0.0


import nltk


def rouge_n(peer, models, n, alpha):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    matches = 0
    recall_total = 0
    peer = peer.split(' ')

    peer_counter = _ngram_counts(peer, n)
    # model do người viết nên có thể có 1 hoặc nhiều bản tóm tắt
    for model in models:
        model = model.split(' ')
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha)


def rouge_1(peer, models, alpha):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 1, alpha)


def rouge_2(peer, models, alpha):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 2, alpha)


def rouge_3(peer, models, alpha):
    """
    Compute the ROUGE-3 (trigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 3, alpha)


def lcs(a, b):
    """
    Compute the length of the longest common subsequence between two sequences.

    Time complexity: O(len(a) * len(b))
    Space complexity: O(min(len(a), len(b)))
    """
    # This is an adaptation of the standard LCS dynamic programming algorithm
    # tweaked for lower memory consumption.
    # Sequence a is laid out along the rows, b along the columns.
    # Minimize number of columns to minimize required memory
    if len(a) < len(b):
        a, b = b, a
    # Sequence b now has the minimum length
    # Quit early if one sequence is empty
    if len(b) == 0:
        return 0
    # Use a single buffer to store the counts for the current row, and
    # overwrite it on each pass
    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    # Return the last cell of the last row
    return left


def rouge_l(peer, models, alpha):
    """
    Compute the ROUGE-L score of a peer with respect to one or more models.
    """
    matches = 0
    recall_total = 0
    for model in models:
        matches += lcs(model, peer)
        recall_total += len(model)
    precision_total = len(models) * len(peer)
    return _safe_f1(matches, recall_total, precision_total, alpha)


start_time = time.time()
import re

if __name__ == "__main__":

    system_path = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn/summaries_lead3'
    #system_path = ROOT_DIR + '/Data_Progress/summaries/centroid_based_1'
    human_path = "/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn/summaries_for_lead"

    arr_rouge_precision = []
    arr_rouge_recall = []
    arr_rouge_f1 = []
    leng = []
    for file in os.listdir(system_path):
        id = file.split('_')[1]
        file_name_system = system_path + '/' + file
        text_system = open(file_name_system, 'r').read().strip().replace("\n", " ").lower()
        text_system = re.sub(r'[.,\-\'`]', ' ', text_system)
        text_system = re.sub(r'\s+', ' ', text_system)
        text_system = text_system.split(' ')

        file_name_human_1 = human_path + '/summari_' + str(id)
        text_human_1 = open(file_name_human_1, 'r').read().strip().replace("\n", " ").lower()
        text_human_1 = re.sub('[.,\-\'`]', ' ', text_human_1)
        text_human_1 = re.sub(r'\s+', ' ', text_human_1)

        precision, recall, f1 = rouge_2(text_system, [text_human_1], 0.5)
        arr_rouge_precision.append(precision)
        arr_rouge_recall.append(recall)
        arr_rouge_f1.append(f1)
        print(f1)

    mean_precision = sum(arr_rouge_precision) / len(arr_rouge_precision)
    mean_recall = sum(arr_rouge_recall) / len(arr_rouge_recall)
    mean_f1 = sum(arr_rouge_f1) / len(arr_rouge_f1)
    print("Rouge-1 với system và human chưa tách từ: ")
    print("%.10s" % ("precision") + " | " + "%.10s" % ("recall") + " | " + "%.10s" % ("F1"))
    print("%.8f" % (mean_precision), "%.7f" % (mean_recall),
          "%.7f" % (mean_f1))
    print("-" * 40)
    print("Execution time: " + str(time.time() - start_time))
