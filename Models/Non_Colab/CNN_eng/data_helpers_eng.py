import re
import os
import itertools
import numpy as np
from collections import Counter
from definitions import ROOT_DIR
from sklearn.model_selection import train_test_split


def write_file_text(data, path_file):
    out = open(path_file, 'w')
    out.write(data)
    out.close()

def load_data_and_labels():
    x_text, y = load_sentences_and_labels()
    x_text = [s.split(" ") for s in x_text]
    y = [[0, 1] if label == 1 else [1, 0] for label in y]
    return [x_text, y]

def load_sentences_and_labels():
    all_sents = []
    path_dir_data = ROOT_DIR + "/Data/CNN/data_labels_clus"

    # get list files train, test
    list_train_files, list_test_files = train_test_split(os.listdir(path_dir_data), test_size= 0.0118, shuffle=True)
    print(len(list_test_files))
    print(len(list_train_files))

    write_file_text('\n'.join(list_test_files), "test_files.txt")
    write_file_text('\n'.join(list_train_files), "train_files.txt")


    for l in [list_train_files, list_test_files]:
        for file in l:
            sents = open(path_dir_data + '/' + file, 'r').read().strip().split('\n')
            all_sents += [s for s in sents if s != '']

    labels = [int(x[0]) for x in all_sents]
    x_text = [a[2:] for a in all_sents]

    return x_text, labels


def load_sentences_and_labels_by_filename():
    all_sents = []
    path_dir_data = "/content/data_labels" #ROOT_DIR + "/data_labels"

    list_train_files = open("train_files.txt").read().split('\n')
    list_test_files = open("test_files.txt").read().split('\n')

    for l in [list_train_files, list_test_files]:
        for clus in l:
            path_clus = path_dir_data + '/' + clus
            for filename in os.listdir(path_clus):
                sents = open(path_clus + '/' + filename, 'r').read().strip().split('\n')
                filename = filename.replace(".txt", '')  # remove txt tail
                all_sents.append((clus + '/' + filename, len(sents), [int(x[0]) for x in sents]))

    return all_sents

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
