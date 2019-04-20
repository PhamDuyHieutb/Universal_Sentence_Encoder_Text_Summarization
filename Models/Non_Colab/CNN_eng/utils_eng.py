import os
import sys
import time
import numpy as np
import pickle
from gensim.models import KeyedVectors
#from definitions import ROOT_DIR
import re
#from Labels_Convert.Utils import text_utils_english

path_data = "/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn_dm/data_labels"


def load_model(model_path):
    # model_path = "baomoi.window2.vn.model.bin"

    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    return word2vec_model


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


def make_vocab():
    vocabulary_list = set()

    list_clus = os.listdir(path_data)

    # clus : train, test, validation
    for clus in list_clus:
        if clus != 'train':
            print(clus)
            all_sents = []
            path_clus = path_data + '/' + clus

            # test and validation folder
            for file in os.listdir(path_clus):
                sents = open(path_clus + '/' + file, 'r').read().strip().split('\n')
                all_sents += [s[2:].strip() for s in sents if s != '']  # remove labels

            all_sents_concat = ' '.join(all_sents)
            all_sents_concat = re.sub(r'\s+', ' ', all_sents_concat)
            vocabulary_list = vocabulary_list.union(all_sents_concat.strip().split(' '))

            print('len vocab', len(vocabulary_list))

    # train folder
    path_train = path_data + '/train'
    for filename in os.listdir(path_train):
        train_sents = []
        docs = open(path_train + '/' + filename, 'r').read().strip().split('\n####\n')
        for doc in docs:
            train_sents += [s[2:] for s in doc.strip().split('\n') if s != '']
        sents_concat = ' '.join(train_sents)
        sents_concat = re.sub(r'\s+', ' ', sents_concat)
        vocabulary_list = vocabulary_list.union(sents_concat.strip().split(' '))

        print('len vocab', len(vocabulary_list))

    vocabulary_list = list(vocabulary_list)
    vocabulary_list.append('<PAD/>')
    with open('words.txt', 'w') as f:
        f.write("\n".join(vocabulary_list))

    return vocabulary_list


def customize_embeddings_from_pretrained_baomoi_w2v(pretrained_embedding_fpath):
    #vocabulary_inv_list = make_vocab()
    vocabulary_inv_list = open('words.txt', 'r').read().strip().split('\n')
    print(len(vocabulary_inv_list))
    vocabulary_inv = {rank: word for rank, word in enumerate(vocabulary_inv_list)}

    directory =  "/home/hieupd/PycharmProjects/CNN_SVM_summarization/Models/Word2vec/models_english"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fpath_pretrained_extracted = os.path.expanduser(
        "{}/cnn_dm_extracted_gnews{}.pl".format(directory, sys.version_info.major))
    fpath_word_list = os.path.expanduser("{}/words.dat".format(directory))

    tic = time.time()
    model = load_model(pretrained_embedding_fpath)
    #loadGloveModel('/home/hieupd/PycharmProjects/Data_DOAN/glove.6B/glove.6B.100d.txt')
    print('Please wait ... (it could take a while to load the file : {})'.format(pretrained_embedding_fpath))
    print('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))

    embedding_weights = {}
    embedding_dim = 300

    found_cnt = 0
    words = []
    not_found = []

    for id, word in vocabulary_inv.items():
        words.append(word)
        if word in model:
            embedding_weights[id] = model[word]
            found_cnt += 1
        elif word.lower() in model:
            embedding_weights[id] = model[word.lower()]
            found_cnt += 1
        else:
            not_found.append(word)
            #embedding_weights[id] = np.random.uniform(-0.25, 0.25, embedding_dim)

    print('found ', found_cnt)
    with open(fpath_pretrained_extracted, 'wb') as f:
        pickle.dump(embedding_weights, f)

    with open(fpath_word_list, 'w') as f:
        f.write("\n".join(words))

    with open('not_found.txt', 'w') as f:
        f.write("\n".join(not_found))


def load_pretrained_embeddings():
    path =  '/home/hieupd/PycharmProjects/CNN_SVM_summarization/Models/Word2vec/models_english/cnn_extracted-python_gnews3.pl'

    with open(path, 'rb') as f:
        embedding_weights = pickle.load(f)

    # embedding_weights is a dictionary {word_index:numpy_array_of_300_dim}

    out = np.array(
        list(embedding_weights.values()))  # added list() to convert dict_values to a list for use in python 3
    # np.random.shuffle(out)

    print('embedding_weights shape:', out.shape)
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    return out


# def fix_words():
#     list_clus = os.listdir(path_data)
#     list_words = open('need_lower.txt', 'r').read().strip().split('\n')
#
#     # f = open('words.txt', 'r')
#     # all_words = f.read().strip()
#     #
#     # for w in list_words:
#     #     all_words = all_words.replace(w, w.lower())
#     # text_utils_english.write_text(all_words.strip(), 'words.txt')
#     for clus in list_clus:
#         if clus == 'test':
#             print(clus)
#             path_clus = path_data + '/' + clus
#
#             # test and validation folder
#             for file in os.listdir(path_clus):
#                 print(file)
#                 path_file = path_clus + '/' + file
#                 f = open(path_file, 'r')
#                 data = f.read()
#                 f.close()
#                 for w in list_words:
#                     if w in data:
#                         data = data.replace(w, w.lower())
#
#                 text_utils_english.write_text(data, path_file)
#
#     # path_train = path_data + '/train'
#     # for filename in os.listdir(path_train):
#     #     print(filename)
#     #     f = open(path_train + '/' + filename, 'r')
#     #     docs = f.read().strip()
#     #     f.close()
#     #     for w in list_words:
#     #         if w in docs:
#     #             docs = docs.replace(w, w.lower())
#     #     text_utils_english.write_text(docs, path_train + '/' + filename)

def main():
    path_to_baomoi_vectors = '/home/hieupd/PycharmProjects/Data_DOAN/GoogleNews-vectors-negative300.bin'

    print('Your path to the baomoi vector file is: ', path_to_baomoi_vectors)
    customize_embeddings_from_pretrained_baomoi_w2v(path_to_baomoi_vectors)


if __name__ == "__main__":
    main()
