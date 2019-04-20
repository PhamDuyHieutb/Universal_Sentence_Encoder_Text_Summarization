from joblib import load
import os
import nltk
import re
from Rouge.rouge_vi.py_rouge_vn_for_svm import rouge_1, rouge_2


def check_rouge(path_summaries, text_system, id):
    text_system = re.sub(r'[.,\-\'`]', ' ', text_system).lower()
    text_system = re.sub(r'\s+', ' ', text_system)
    text_system = text_system.split(' ')

    file_name_human_1 = path_summaries + '/summari_' + str(id)
    text_human_1 = open(file_name_human_1, 'r').read().strip().replace("\n", " ").lower()
    text_human_1 = re.sub('[.,\-\'`]', ' ', text_human_1)
    text_human_1 = re.sub(r'\s+', ' ', text_human_1)

    precision, recall, f1 = rouge_1(text_system, [text_human_1], 0.5)

    return f1


def check_lead_based_rouge(path_summaries, pathfile, n):
    id = pathfile.split('_')[-1]
    doc = open(pathfile, 'r').read().strip()

    arr_all_sents = nltk.sent_tokenize(doc)
    arr_all_sents = [re.sub(r'[.,\-\'`]', ' ', sent) for sent in arr_all_sents]

    text_system = ' '.join(arr_all_sents[:3]).lower()

    text_system = re.sub(r'\s+', ' ', text_system)
    text_system = text_system.split(' ')

    file_name_human_1 = path_summaries + '/summari_' + str(id)
    text_human_1 = open(file_name_human_1, 'r').read().strip().replace("\n", " ").lower()
    text_human_1 = re.sub('[.,\-\'`]', ' ', text_human_1)
    text_human_1 = re.sub(r'\s+', ' ', text_human_1)

    if n == 1:
        precision, recall, f1 = rouge_1(text_system, [text_human_1], 0.5)
    else:
        precision, recall, f1 = rouge_2(text_system, [text_human_1], 0.5)

    return f1


if __name__ == '__main__':

    root_data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data'
    cnn_root = root_data + '/cnn'


    # path_data = cnn_root + '/documents_for_lead'
    # path_results = cnn_root + '/summaries_lead3'

    path_data = cnn_root + '/test/doc_test'
    path_results = cnn_root + '/test/sum_lead3'

    if not os.path.exists(path_results):
        os.mkdir(path_results)

    count = 0
    list_files = []
    for name_file in os.listdir(path_data):
        id = name_file.split('_')[1]
        doc = open(path_data + '/' + name_file, 'r').read().strip()

        arr_all_sents = nltk.sent_tokenize(doc)
        #arr_all_sents = [re.sub(r'[\'`]', '', sent) for sent in arr_all_sents]

        f = open(path_results + '/system_' + id, 'w')
        f.write('\n'.join(arr_all_sents[:3]))
