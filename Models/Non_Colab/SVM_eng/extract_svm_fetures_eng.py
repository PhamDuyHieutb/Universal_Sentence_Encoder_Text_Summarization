from sklearn.feature_extraction.text import TfidfVectorizer
from Models.Non_Colab.SVM_eng import text_utils_eng
from definitions import ROOT_DIR
import os




SPECICAL_CHARACTER = {'(', ')', '[', ']', '"'}


class FeatureSvm(object):
    def __init__(self, file_name, file_stopwords):
        self.file_name = file_name
        self.stop_words = text_utils_eng.read_stopwords(file_stopwords)

    def get_sentences(self):
        sentences = text_utils_eng.split_sentences(self.file_name)
        return text_utils_eng.text_process(sentences, self.stop_words)

    # 1/Position
    def extract_feature1(self, sentence, sentences):
        return 1.0 / (sentences.index(sentence) + 1)

    # Doc_First
    def extract_feature2(self, sentence):
        contain_first_paragraph = text_utils_eng.get_sentence_first_paragraph(self.file_name, self.stop_words)

        if sentence in contain_first_paragraph:
            return 1
        else:
            return 0

    # Length
    def extract_feature3(self, sentence):
        words = []
        for item in sentence.split(' '):
            if item not in SPECICAL_CHARACTER:
                words.append(item)

        return len(words)

    # Quote
    def extract_feature4(self, sentence):
        words = []
        for item in sentence.split(' '):
            if item in SPECICAL_CHARACTER:
                words.append(item)

        return len(words)

    # FreqWord_Uni, FreqWord_Bi
    def extract_feature7_8(self, sentence, document):
        freq_words = text_utils_eng.get_freq_word_uni(document)

        feature = 0.0
        for item in sentence.split(' '):
            if item not in freq_words:
                continue
            else:

                feature += freq_words[item]

        return feature

    # Centroid_Uni, Centroid_Bi
    def extract_feature5_6(self, sentence, document, all_idf):

        centroid_uni = text_utils_eng.get_centroid_uni(document, all_idf)

        feature = 0.0
        for item in sentence.split(' '):
            if item not in centroid_uni:
                continue
            else:
                feature += centroid_uni[item]

        return feature

    # FirstRel_Doc
    def extract_feature9(self, sentence, all_idf, sentences):
        return text_utils_eng.cos_similarity(sentence, all_idf, sentences)

    def get_all_feature_from_sentence(self, sentence, sentences, all_idf, bi_sentence, bi_all_idf):
        document = text_utils_eng.get_doc_from_sentences(sentences)
        bi_document = text_utils_eng.convert_uni_to_bi([document])[0]


        feature1 = self.extract_feature1(sentence, sentences)
        feature2 = self.extract_feature2(sentence)
        feature3 = self.extract_feature3(sentence)
        feature4 = self.extract_feature4(sentence)
        feature5 = self.extract_feature5_6(sentence, document, all_idf)
        feature6 = self.extract_feature5_6(bi_sentence, bi_document, bi_all_idf)
        feature7 = self.extract_feature7_8(sentence, document)
        feature8 = self.extract_feature7_8(bi_sentence, bi_document)
        feature9 = self.extract_feature9(sentence, all_idf, sentences)

        return [feature1, feature2, feature3, feature5, feature6, feature7, feature8, feature9]

    def get_all_feature_from_doc(self, sentences, all_idf, bi_all_idf):

        features = []
        for item in sentences:
            bi_sentence = text_utils_eng.convert_uni_to_bi([item])[0]

            feature = self.get_all_feature_from_sentence(item, sentences, all_idf, bi_sentence, bi_all_idf)
            features.append(feature)

        return features

if __name__ == "__main__":

    path_stopword = 'stopwords_eng.txt'
    stop_words = text_utils_eng.read_stopwords(path_stopword)
    root = ROOT_DIR + '/Data/CNN/data_labels_clus/train/'
    #SVM_FEATURES = ROOT_DIR + '/Data/cnn_sub/svm_features'

    list_document_paths = []

    for filename in os.listdir(root):
        list_document_paths.append(root + filename)

    list_train_files = []
    for filename in os.listdir(root):
        list_train_files.append(root + filename)


    train_documents = text_utils_eng.read_all_train_documents(list_train_files, stop_words)

    print("create all idf")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_documents)
    idf = vectorizer.idf_
    all_idf = dict(zip(vectorizer.get_feature_names(), idf))

    text_utils_eng.save_idf(all_idf, 'all_idf.json')


    bi_documents = text_utils_eng.convert_uni_to_bi(train_documents)
    print("create all bi idf")
    bi_all_idf = text_utils_eng.get_all_idf(bi_documents)
    text_utils_eng.save_idf(bi_all_idf, 'all_bi_idf.json')


    for doc_path in list_document_paths:
        print(doc_path)
        spl_doc = doc_path.split('/')
        name_doc = spl_doc[-1]

        # initial extract feature
        extract_feature = FeatureSvm(doc_path, path_stopword)

        sentences = extract_feature.get_sentences() # be lowered

        # get list label of each file
        labels, sents_nolabel = text_utils_eng.separate_label_sent(sentences)

        arr_features_svm = extract_feature.get_all_feature_from_doc(sents_nolabel, all_idf, bi_all_idf)
        arr_svm_normal = text_utils_eng.normalize(arr_features_svm)

        text_utils_eng.prepare_data_svm(labels, arr_svm_normal, SVM_FEATURES + '/' + name_doc)
