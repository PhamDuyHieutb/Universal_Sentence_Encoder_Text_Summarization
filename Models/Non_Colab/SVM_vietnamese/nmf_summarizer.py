import nltk
import numpy as np
from pyvi import ViTokenizer
from sklearn.decomposition import NMF


class sentence(object):

    def __init__(self, stemmedWords):

        self.stemmedWords = stemmedWords
        self.wordFrequencies = self.sentenceWordFreqs()

    def getStemmedWords(self):
        return self.stemmedWords

    def getWordFreqs(self):
        return self.wordFrequencies

    def sentenceWordFreqs(self):
        wordFreqs = {}
        for word in self.stemmedWords:
            if word not in wordFreqs.keys():
                wordFreqs[word] = 1
            else:
                wordFreqs[word] = wordFreqs[word] + 1

        return wordFreqs


def processFileVietNamese(documents):

    sentences = []

    for sent in documents:
        sentences.append(sentence(sent))

    return sentences


def normalize(numbers):
    max_number = max(numbers)
    normalized_numbers = []

    for number in numbers:
        normalized_numbers.append(number / max_number)

    return normalized_numbers


def getNMF(documents):
    sentences = processFileVietNamese(documents)
    # tf
    vocabulary = []
    for sent in sentences:
        vocabulary = vocabulary + sent.getStemmedWords()
    vocabulary = list(set(vocabulary))
    A = np.zeros(shape=(len(vocabulary), len(sentences)))
    for i in range(len(sentences)):
        tf_sentence = sentences[i].getWordFreqs()
        for word in tf_sentence.keys():
            index = vocabulary.index(word)
            A[index][i] += tf_sentence[word]

    rank_A = np.linalg.matrix_rank(A)
    model = NMF(n_components=rank_A, init='random', random_state=0)
    W = model.fit_transform(A)
    H = model.components_
    scores = np.sum(H, axis=0)
    return normalize(list(scores))
