import os
import sys
import time
import numpy as np
import pickle
from Models.CNN import data_helpers
from gensim.models import KeyedVectors
from definitions import ROOT_DIR

def load_model(model_path):
    #model_path = "baomoi.window2.vn.model.bin"

    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)


    return word2vec_model



def customize_embeddings_from_pretrained_baomoi_w2v(pretrained_embedding_fpath):
    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {rank: word for rank, word in enumerate(vocabulary_inv_list)}
    embedding_dim = 300

    directory =  ROOT_DIR + "/Models/Word2vec/models_vietnamese"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fpath_pretrained_extracted = os.path.expanduser("{}/baomoi_extracted-python{}.pl".format(directory, sys.version_info.major))
    fpath_word_list = os.path.expanduser("{}/words.dat".format(directory))

    tic = time.time()
    model = load_model(pretrained_embedding_fpath)
    print('Please wait ... (it could take a while to load the file : {})'.format(pretrained_embedding_fpath))
    print('Done.  (time used: {:.1f}s)\n'.format(time.time()-tic))

    embedding_weights = {}

    found_cnt = 0
    words = []
    for id, word in vocabulary_inv.items():
        words.append(word)
        if word in model:
            embedding_weights[id] = model[word]
            found_cnt += 1
        else:
            embedding_weights[id] = np.random.uniform(-0.25, 0.25, embedding_dim)
    with open(fpath_pretrained_extracted, 'wb') as f:
        pickle.dump(embedding_weights, f)
    with open(fpath_word_list, 'w') as f:
        f.write("\n".join(words))

def load_pretrained_embeddings():
    path = ROOT_DIR + '/Models/Word2vec/models_vietnamese/baomoi_extracted-python3.pl'

    with open(path, 'rb') as f:
        embedding_weights = pickle.load(f)

    # embedding_weights is a dictionary {word_index:numpy_array_of_300_dim}

    out = np.array(
        list(embedding_weights.values()))  # added list() to convert dict_values to a list for use in python 3
    # np.random.shuffle(out)

    print('embedding_weights shape:', out.shape)
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    return out


def main():

    path_to_baomoi_vectors = ROOT_DIR + '/Models/Word2vec/baomoi.window2.vn.model.bin'

    print('Your path to the baomoi vector file is: ', path_to_baomoi_vectors)
    customize_embeddings_from_pretrained_baomoi_w2v(path_to_baomoi_vectors)


if __name__ == "__main__":
    main()
