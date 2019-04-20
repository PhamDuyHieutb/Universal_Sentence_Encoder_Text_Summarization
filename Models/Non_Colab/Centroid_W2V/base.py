"""
    Author: Gaetano Rossiello
    Email: gaetano.rossiello@uniba.it
"""
import re
import string
import unidecode
import numpy as np
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from gensim.summarization.textcleaner import split_sentences as gensim_sent_tokenize
import flashtext
from definitions import ROOT_DIR


def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = 1 - cosine(v1, v2)
    return score


class BaseSummarizer:

    extra_stopwords = ["''", "``", "'s"]

    def __init__(self,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=10,
                 debug=False):
        self.language = language
        self.preprocess_type = preprocess_type
        self.stopwords_remove = stopwords_remove
        self.length_limit = length_limit
        self.debug = debug

        return

    def sent_tokenize(self, text):
        if self.preprocess_type == 'nltk':
            sents = nltk_sent_tokenize(text, self.language)
        else:
            sents = gensim_sent_tokenize(text)
        sents_filtered = []
        for s in sents:
            if s[-1] != ':' and len(s) > self.length_limit:
                sents_filtered.append(s)
            # else:
            #   print("REMOVED!!!!" + s)
        return sents_filtered

    def read_text_files(self, path):
        f = open(path, 'r').read().strip()

        return f

    def remove_stopwords(self, sent, stopwords):
        new_token = []
        sent = sent.strip().lower()
        for w in sent.split(' '):
            if w not in stopwords:
                new_token.append(w)

        return new_token

    def preprocess_text_nltk(self, text):
        sentences = self.sent_tokenize(text)

        sentences_cleaned = []

        list_stopwords = self.read_text_files(ROOT_DIR + "/Models/Non_Colab/SVM_eng/stopwords_eng.txt").split('\n')

        for sent in sentences:
            token = []
            if self.stopwords_remove:
                token += self.remove_stopwords(sent, list_stopwords)

            words = [w for w in token if w not in string.punctuation]
            words = [w for w in words if w not in self.extra_stopwords]

            if len(words) > 1:
                sentences_cleaned.append(" ".join(words))
            else:
                sentences_cleaned.append('')
        return sentences_cleaned

    def preprocess_text_regexp(self, text):
        sentences = self.sent_tokenize(text)
        sentences_cleaned = []

        list_stopwords = self.read_text_files(ROOT_DIR + "/Models/Non_Colab/SVM_eng/stopwords_eng.txt").split('\n')

        for sent in sentences:
            sent_ascii = unidecode.unidecode(sent)
            cleaned_text = re.sub("[^a-zA-Z0-9]", " ", sent_ascii)
            token = []
            if self.stopwords_remove:
                token += self.remove_stopwords(sent, list_stopwords)

            #words = cleaned_text.lower().split()
            sentences_cleaned.append(" ".join(token))
        return sentences_cleaned

    def preprocess_text(self, text):
        if self.preprocess_type == 'nltk':
            return self.preprocess_text_nltk(text)
        else:
            return self.preprocess_text_regexp(text)

    def summarize(self, text, limit_type='word', limit=100):
        raise NotImplementedError("Abstract method")