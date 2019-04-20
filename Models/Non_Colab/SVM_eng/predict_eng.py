from joblib import load
import numpy as np
from definitions import ROOT_DIR
from Models.Non_Colab.SVM_eng import text_utils_eng
from Models.Non_Colab.Centroid_W2V import mmr_summarizer
from Models.Non_Colab.SVM_eng import mmr_selection_eng
import re
import os
import math



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


def normalize(ar_numbers):
    # normalize array numbers
    probs = []
    for i in range(len(ar_numbers)):
        probs.append(ar_numbers[i][1])
    max_number = max(probs)
    normalized_numbers = []
    for number in probs:
        normalized_numbers.append(number / max_number)

    return normalized_numbers


if __name__ == '__main__':

    path_data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'
    path_prob_uni = path_data + '/test_prob_3'
    path_results = ROOT_DIR + "/Data_Progress/summaries/universal_3"
    f_test = []

    list_test_files = os.listdir(path_prob_uni)
    idf = text_utils_eng.read_json_file('all_idf.json')

    list_stopwords = text_utils_eng.read_file_text(ROOT_DIR + "/Models/Non_Colab/SVM_eng/stopwords_eng.txt").strip().split('\n')

    for name_file in list_test_files:
        print('file', name_file)
        id = name_file.replace('.npy', '').split('_')[1]

        prob_predic = np.load(path_prob_uni + '/' + name_file)
        prob_predic = sorted(enumerate(prob_predic), key=lambda x: list(x[1])[1], reverse=True)


        doc = open(path_data + '/data_labels/test/' + name_file.replace('.npy', ''), 'r').read().strip()

        arr_all_sents = doc.split("\n")
        print( 'prob', len(prob_predic), 'all sent', len(arr_all_sents))
        sents_values = []
        for s in prob_predic:
            index = int(s[0])
            posi_fea = math.log1p(1/(1 + index))
            sent = arr_all_sents[index][2:]
            ele = (index, sent, text_utils_eng.clean_stem_sent(sent,list_stopwords),s[1][1], posi_fea)
            sents_values.append(ele)

        ###################################### test

        sents_values = sorted(sents_values, key=lambda x: x[0])

        summari = mmr_selection_eng.make_summary(sents_values, 0.9)  # mmr_summarizer.getMMR(list_sents) #

        # summari = re.sub(r'[`]', '', summari)
        # summari = summari.replace("''", '')

        summari = re.sub(r'\s+', ' ', summari)

        f = open(path_results + '/system_' + id, 'w')
        f.write(summari)