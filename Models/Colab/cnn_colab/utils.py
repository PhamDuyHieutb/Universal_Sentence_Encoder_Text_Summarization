import os
import sys
import time
import numpy as np
import pickle
import itertools
from collections import Counter
from gensim.models import KeyedVectors
from definitions import ROOT_DIR

def load_model(model_path):
    #model_path = "baomoi.window2.vn.model.bin"

    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)


    return word2vec_model


def load_data_and_labels():
    x_text, y = load_sentences_and_labels()
    x_text = [s.split(" ") for s in x_text]
    y = [[0, 1] if label == 1 else [1, 0] for label in y]
    return [x_text, y]


def load_sentences_and_labels():
    all_sents = []
    path_dir_data =  ROOT_DIR + '/2k_data_labels'   #"/content/gdrive/My Drive/20182_DOAN/data_labels"
    for clus in os.listdir(path_dir_data):
        path_clus = path_dir_data + '/' + clus
        for filename in os.listdir(path_clus):
            sents = open(path_clus + '/' + filename, 'r').read().strip().split('\n')
            all_sents += sents

    labels = [int(x[0]) for x in all_sents]
    x_text = [a[2:] for a in all_sents]

    return x_text, labels


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    y = y.argmax(axis=1)

    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()


    sentences_padded = pad_sentences(sentences)
    # vocabulary is a dict
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]



def customize_embeddings_from_pretrained_baomoi_w2v(pretrained_embedding_fpath):
    x, y, vocabulary, vocabulary_inv_list = load_data()
    vocabulary_inv = {rank: word for rank, word in enumerate(vocabulary_inv_list)}
    embedding_dim = 300

    directory = ROOT_DIR + "/Models/Word2vec/models_vietnamese"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fpath_pretrained_extracted = os.path.expanduser("{}/200clusters_extracted-python{}.pl".format(directory, sys.version_info.major))
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

def main():

    path_to_baomoi_vectors = ROOT_DIR + '/Models/Word2vec/baomoi.window2.vn.model.bin'

    print('Your path to the baomoi vector file is: ', path_to_baomoi_vectors)
    customize_embeddings_from_pretrained_baomoi_w2v(path_to_baomoi_vectors)

if __name__ == "__main__":
    main()
