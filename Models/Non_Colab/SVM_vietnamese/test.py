import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import nltk
import os
from definitions import ROOT_DIR
# with open("w2v.model", 'rb') as f:
#     model_w2v = pickle.load(f)
#
# def caculateVector(sentences, output):
#     all_sent = []
#     for sent in sentences:
#         result = np.zeros(300)
#         for word in sent.split(' '):
#             try:
#                 result += np.array(model_w2v[word])
#             except:
#                 continue
#         all_sent.append(result)
#         np.save(output, all_sent)
#     return all_sent
#
#
# if __name__ == '__main__':
#
#     path = '/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/data_labels/train'
#     path_output_w2v = '/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/w2v/train'
#     for clus in os.listdir(path):
#         print(clus)
#         list_sentences = []
#         path_clus = path + '/' + clus
#         for file_name in os.listdir(path_clus):
#             sents = open(path_clus + '/' + file_name, 'r').read().split('\n')
#             for s in sents:
#                 list_sentences.append(s[2:])
#
#
#         caculateVector(list_sentences, path_output_w2v + '/' + clus)


# devide vectors to their file and clusters

path_data = '/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/Data/data_labels'


def get_cnn_features():
    X = np.load("/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/cnn_sent_embed/sentence_vectors.npy")
    Y = np.load("/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/cnn_sent_embed/sentence_vectors_y.npy")

    pathout = '/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/cnn_features/'
    pivot = 0

    for clus in os.listdir(path_data):
        path_clus = path_data + '/' + clus
        clus_vectors = []
        all_sents = []
        for filename in os.listdir(path_clus):
            all_sents += list(open(path_clus + '/' + filename, 'r').readlines())

        num_sents = len(all_sents)
        vectors = X[pivot:(pivot + num_sents)]
        labels = Y[pivot:(pivot + num_sents)]
        labels_test = [int(a[0]) for a in all_sents]
        pivot += num_sents
        if np.array_equal(labels, labels_test):
            np.save(pathout + clus , vectors)
        else:
            print(labels_test)
            print(labels)


    # for clus in os.listdir(path_data):
    #     path_clus = path_data + '/' + clus
    #     clus_vectors = []
    #     all_sents = []
    #     for filename in os.listdir(path_clus):
    #         all_sents += list(open(path_clus + '/' + filename, 'r').readlines())
    #
    #     num_sents = len(all_sents)
    #     vectors = X[pivot:(pivot + num_sents)]
    #     labels = Y[pivot:(pivot + num_sents)]
    #     labels_test = [int(a[0]) for a in all_sents]
    #     pivot += num_sents
    #     if np.array_equal(labels, labels_test):
    #         np.save(pathout + clus , vectors)
    #     else:
    #         print(labels_test)
    #         print(labels)

from sklearn.preprocessing import normalize
import math
import shutil
if __name__ == '__main__':
    pass