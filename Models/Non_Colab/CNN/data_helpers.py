import re
import os
import itertools
import numpy as np
from collections import Counter
from definitions import ROOT_DIR

SPECICAL_CHARACTER = {'(', ')', '[', ']', '"', '”', '!', '?', '“', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                      '9'}


def clean_str(string):
    # tmp = ViTokenizer.tokenize(string).lower()  # with data need to lower and token
    # string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    text_tmp = []
    for word in string.split(' '):
        if len(word) != 1 or word in SPECICAL_CHARACTER:
            text_tmp.append(word)

    return ' '.join(text_tmp).strip()


def load_data_and_labels():
    x_text, y = load_sentences_and_labels()
    x_text = [s.split(" ") for s in x_text]
    y = [[0, 1] if label == 1 else [1, 0] for label in y]
    return [x_text, y]


def load_sentences_and_labels():
    all_sents = []
    path_dir_data = ROOT_DIR + "/data_labels"

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
