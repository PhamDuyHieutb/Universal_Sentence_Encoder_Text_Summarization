"""
Implementation based on paper:

Centroid-based Text Summarization through Compositionality of Word Embeddings

Author: Gaetano Rossiello
Email: gaetano.rossiello@uniba.it
"""
from Models.Non_Colab.Centroid_W2V import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import KeyedVectors
import math
from Models.Non_Colab.SVM_eng import mmr_selection_eng, text_utils_eng
from Models.Non_Colab.Centroid_W2V import mmr_summarizer
from definitions import ROOT_DIR
import operator
from rake_nltk import Rake

def average_score(scores):
    # average of position, centroid closest, length of sent
    score = 0
    count = 0
    for s in scores:
        if s > 0:
            score += s
            count += 1
    if count > 0:
        score /= count
        return score
    else:
        return 0


def stanford_cerainty_factor(scores):
    score = 0
    minim = 100000
    for s in scores:
        score += s
        if s < minim & s > 0:
            minim = s
    score /= (1 - minim)
    return score


def get_max_length(sentences):
    max_length = 0
    for s in sentences:
        l = len(s.split())
        if l > max_length:
            max_length = l
    return max_length


def load_gensim_embedding_model(model_path):
    # available_models = gensim_data_downloader.info()['models'].keys()
    # assert model_name in available_models, 'Invalid model_name: {}. Choose one from {}'.format(model_name, ', '.join(available_models))
    # model_path = gensim_data_downloader.load(model_name, return_path=True)
    return KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')


class CentroidWordEmbeddingsSummarizer(base.BaseSummarizer):
    def __init__(self,
                 embedding_model,
                 language='english',
                 preprocess_type='nltk',
                 stopwords_remove=True,
                 length_limit=5,
                 debug=False,
                 topic_threshold=0.35,
                 sim_threshold=0.95,
                 reordering=True,
                 zero_center_embeddings=False,
                 keep_first=False,
                 bow_param=0,
                 length_param=0,
                 position_param=0):
        super().__init__(language, preprocess_type, stopwords_remove, length_limit, debug)

        self.embedding_model = embedding_model

        self.word_vectors = dict()

        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        self.reordering = reordering

        self.keep_first = keep_first
        self.bow_param = bow_param
        self.length_param = length_param
        self.position_param = position_param

        self.zero_center_embeddings = zero_center_embeddings

        if zero_center_embeddings:
            self._zero_center_embedding_coordinates()
        return

    def get_bow(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, centroid_vector

    def get_list_centroid_word(self, sentences):

        # get controid word list
        # get sum tfidf of each word in all docs/sents => normalize => filter with threshold

        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(axis=0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())

        feature_names = vectorizer.get_feature_names()

        relevant_vector_indices = np.where(centroid_vector > self.topic_threshold)[0]

        word_list = list(np.array(feature_names)[relevant_vector_indices])[:10]
        print(word_list)
        return word_list

    # def get_list_centroid_word(self, sentences, idf):
    #
    #     # get controid word list
    #     # get sum tfidf of each word in all docs/sents => normalize => filter with threshold
    #     dict_words = text_utils_eng.get_dict_words_from_doc(' '.join(sentences))
    #
    #     vectorizer = CountVectorizer()
    #     sent_word_matrix = vectorizer.fit_transform(sentences)
    #     feature_names = vectorizer.get_feature_names()
    #     list_word_tfidf = {}
    #     for word in feature_names:
    #         tf_word = text_utils_eng.get_word_freq(word, dict_words)
    #         if word in idf:
    #             idf_w = idf[word]
    #         else:
    #             idf_w = 0.1
    #         list_word_tfidf[word] = tf_word * idf_w
    #     print(sorted(list_word_tfidf.items(), key=operator.itemgetter(1), reverse= True))
    #     l_sorted = sorted(list_word_tfidf, key=list_word_tfidf.get, reverse=True)
    #     centroid_words = l_sorted[:8]
    #
    #
    #     return centroid_words

    def word_vectors_cache(self, sentences):
        self.word_vectors = dict()
        for s in sentences:
            words = s.replace('.', '').split()
            for w in words:
                if self.word_vectors.get(w) is None:
                    if w.lower() in self.embedding_model:
                        self.word_vectors[w] = self.embedding_model[w.lower()]
                    else:
                        self.word_vectors[w] = np.random.uniform(-0.25, 0.25, 50)

        return

    # Sentence representation with sum of word vectors
    def compose_vectors(self, words):
        # composed_vector = np.zeros(self.embedding_model.vector_size, dtype="float32")
        composed_vector = np.zeros(50, dtype="float32")
        word_vectors_keys = set(self.word_vectors.keys())
        count = 0
        for w in words:
            if w in word_vectors_keys:
                composed_vector = composed_vector + self.word_vectors[w]
                count += 1

        if count != 0:
            composed_vector = np.divide(composed_vector, count)
        return composed_vector

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences_full = self.preprocess_text(text)  # remove stopwords, lower, still keep num sents

        clean_sentences = [sent for sent in clean_sentences_full if sent != '']

        if self.debug:
            print("ORIGINAL TEXT STATS = {0} chars, {1} words, {2} sentences".format(len(text),
                                                                                     len(text.split(' ')),
                                                                                     len(raw_sentences)))
            print("*** RAW SENTENCES ***")
            for i, s in enumerate(raw_sentences):
                print(i, s)
            print("*** CLEAN SENTENCES ***")
            for i, s in enumerate(clean_sentences):
                print(i, s)

        idf = text_utils_eng.read_json_file(ROOT_DIR + '/Models/Non_Colab/SVM_eng/all_idf.json')
        centroid_words = self.get_list_centroid_word(clean_sentences)

        if self.debug:
            print("*** CENTROID WORDS ***")
            print(len(centroid_words), centroid_words)

        # prepare word embed vector
        self.word_vectors_cache(clean_sentences)

        # get centroid vector w2v
        centroid_vector = self.compose_vectors(centroid_words)

        # get centroid bag of words
        tfidf, centroid_bow = self.get_bow(clean_sentences)
        max_length = get_max_length(clean_sentences)

        sentences_scores = []
        for i in range(len(clean_sentences_full)):
            sent = clean_sentences_full[i]

            if sent != '':
                ele = {}
                scores = []
                words = sent.split()

                # get sent vector from word embedding
                sentence_vector = self.compose_vectors(words)
                #sentence_vector = embedd_sents[i]

                # compute score by similarity with centroid vector w2v, centroid vector with bag of words, leng sent, position
                scores.append(base.similarity(sentence_vector, centroid_vector))
                #scores.append(self.bow_param * base.similarity(tfidf[i, :], centroid_bow))
                scores.append(self.length_param * (1 - (len(words) / max_length)))
                scores.append(self.position_param * math.log1p(1 / (i + 1)))

                score = average_score(scores)
                ele['score'] = score
                ele['origin'] = raw_sentences[i]
                ele['clean']  = clean_sentences_full[i]

                sentences_scores.append(ele)
                if self.debug:
                    print(i, scores, score)


        doc = ' '.join(clean_sentences)
        sentences_summary = mmr_selection_eng.make_summary(sentences_scores, 0.85, idf, doc)

        # # if wanna get first sentence as default
        # if self.keep_first:
        #     for s in sentence_scores_sort:
        #         if s[0] == 0:
        #             sentences_summary.append(s)
        #             if limit_type == 'word':
        #                 count += len(s[1].split())
        #             else:
        #                 count += len(s[1])
        #             sentence_scores_sort.remove(s)
        #             break

        # for s in sentence_scores_sort:
        #     if count > limit:
        #         break
        #     include_flag = True
        #     for ps in sentences_summary:
        #         sim = base.similarity(s[3], ps[3])
        #         # print(s[0], ps[0], sim)
        #         if sim > self.sim_threshold:
        #             include_flag = False
        #     if include_flag:
        #         # print(s[0], s[1])
        #         sentences_summary.append(s)
        #         if limit_type == 'word':
        #             count += len(s[1].split())
        #         else:
        #             count += len(s[1])
        #
        # if self.reordering:
        #     sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

        summary = "\n".join([s['origin'] for s in sentences_summary])

        if self.debug:
            print("SUMMARY TEXT STATS = {0} chars, {1} words, {2} sentences".format(len(summary),
                                                                                    len(summary.split(' ')),
                                                                                    len(sentences_summary)))

            print("*** SUMMARY ***")
            print(summary)

        return summary

    def _zero_center_embedding_coordinates(self):
        # Create the centroid vector of the whole vector space
        count = 0
        self.centroid_space = np.zeros(self.embedding_model.vector_size, dtype="float32")
        self.index2word_set = set(self.embedding_model.wv.index2word)
        for w in self.index2word_set:
            self.centroid_space = self.centroid_space + self.embedding_model[w]
            count += 1
        if count != 0:
            self.centroid_space = np.divide(self.centroid_space, count)
