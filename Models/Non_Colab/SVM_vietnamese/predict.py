from joblib import load
from Models.SVM_vietnamese import text_utils
import numpy as np
import os
from definitions import ROOT_DIR
from pyvi import ViTokenizer
from Models.SVM_vietnamese import mmr_selection

np.set_printoptions(threshold=np.inf)


def get_doc_id(index, leng_docs):
    '''

    :param index: index of sent in array all sentences
    :param leng_docs: is a dict has num sent of each doc
    :return:
    '''
    sum_len = 0
    id = 0
    for i in leng_docs:
        if index < sum_len + leng_docs[i]:
            id = i
            break
        else:
            sum_len += leng_docs[i]

    return id


def evaluate_prob(X_test):
    # predict document input

    reload = load("svm_model")
    predict = reload.predict_proba(X_test)
    sort_predict = sorted(enumerate(predict), key=lambda x: list(x[1])[0], reverse=True)
    return sort_predict


if __name__ == '__main__':

    path_cnn_features = ROOT_DIR + '/Data_Progress/cnn_features/'
    path_data = ROOT_DIR + "/Data_Progress/data_labels/"
    #path_svm_features = ROOT_DIR + "/Data_Progress/svm_features/"
    path_results = ROOT_DIR + "/Data_Progress/results/"
    f_test = []

    list_test_files = open( ROOT_DIR + "/Labels_Convert/Filters/test_files.txt").read().split("\n")

    # for filename in list_test_files:
    #     filename = "/".join(filename.split("/")[-2:])
    #
    #     t = open(path_svm_features + filename, 'r')
    #     f_test.append((filename, t.read().split('\n')))
    #     t.close()

    #idf = text_utils.read_json_file('all_idf.json')

    # for name_file, svm_features in f_test:
    #     print('file', name_file)
    #
    #     clus = name_file.split("/")[0]
    #
    #     if not os.path.exists(path_results + clus):
    #         os.mkdir(path_results + clus)
    #
    #     X_test = []
    #     Y_test = []
    #     for fea in svm_features:
    #         X_test.append(list(map(float, fea[2:].split(' '))))  # [:9]
    #         Y_test.append(int(fea[0]))
    #
    #     X_test = np.array(X_test)
    #
    #     cnn_features = np.load(path_cnn_features + name_file.replace("txt", 'npy'))
    #
    #     X_test = np.concatenate((X_test, cnn_features), axis=1)
    #     prob_predic = evaluate_prob(X_test)
    #
    #     human_path = ROOT_DIR + "/Data_Progress/baomoi_2k/summaries/"
    #     file_human_1 = human_path + name_file
    #     text_1 = open(file_human_1, 'r').read()
    #     text_1_token = ViTokenizer.tokenize(text_1)
    #     length_summary = len(text_1_token.split())
    #
    #     doc = open(path_data + name_file, 'r').read()
    #
    #     arr_all_sents = doc.split("\n")
    #
    #     sen_origin = []
    #     for s in prob_predic:
    #         ele = {}
    #         index = int(s[0])
    #         ele['value'] = arr_all_sents[index][2:].lower()
    #         ele['score_svm'] = list(s[1])[0]
    #         sen_origin.append(ele)
    #
    #     summari = mmr_selection.make_summary(sen_origin, length_summary, 0.85, idf, doc)
    #
    #     summari_sents = [sent['value'] for sent in summari]
    #
    #     f = open(path_results + name_file, 'w')
    #     f.write('\n'.join(summari_sents))


    for name_file in list_test_files:
        print('file', name_file)


        X_test = np.load(path_cnn_features + name_file.replace("story", 'npy'))

        prob_predic = evaluate_prob(X_test)

        human_path = ROOT_DIR + "/Data/CNN/summaries"
        file_human_1 = human_path + '/' + name_file
        text_1 = open(file_human_1, 'r').read()
        text_1_token = ViTokenizer.tokenize(text_1)
        length_summary = len(text_1_token.split())

        doc = open(path_data + name_file, 'r').read()

        arr_all_sents = doc.split("\n")

        sen_origin = []
        for s in prob_predic:
            ele = {}
            index = int(s[0])
            ele['value'] = arr_all_sents[index][2:].lower()
            ele['score_svm'] = list(s[1])[0]
            sen_origin.append(ele)

        summari = mmr_selection.make_summary(sen_origin, length_summary, 0.85, idf, doc)

        summari_sents = [sent['value'] for sent in summari]

        f = open(path_results + name_file, 'w')
        f.write('\n'.join(summari_sents))
