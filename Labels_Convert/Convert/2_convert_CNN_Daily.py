from Rouge.rouge_vi import py_rouge_vn_for_svm
from Labels_Convert.Utils import text_utils_english
import os
import nltk

porter = nltk.PorterStemmer()


class ConvertExtract(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def convert_extract_with_indexes(self, list_documents, list_human):
        sentences_system, docs_reference = \
            text_utils_english.get_all_sentences(list_documents, list_human)

        old_rouge = 0
        rouge = 0  # initial rouge score
        old_index = -1

        sentences_label = []
        arr_indexes = []
        all_sentences = []

        # add all list sentences of system to all_sentences in a cluster
        for filename, list_sents in sentences_system:
            all_sentences += list_sents

        all_sents_steamed = []
        for sent in all_sentences:
            all_sents_steamed.append(' '.join([porter.stem(word) for word in sent.split(' ')]))

        sentences_selected = []  # arr sentence is choosed

        # with each round with add new sentence
        while (0 == 0):
            i = 0

            for sent in all_sents_steamed:  # SELECT sentence which has best rouge, return index of this sentence
                tmp = []
                for s in sentences_selected:
                    tmp.append(s)

                tmp.append(sent)
                tmp = ' '.join(tmp)
                # Use rouge 1
                tmp_rouge_f1 = py_rouge_vn_for_svm.rouge_1(tmp, docs_reference, self.alpha)

                if tmp_rouge_f1 > rouge:  # if has change score
                    rouge = tmp_rouge_f1
                    old_index = i

                i += 1

            if rouge == old_rouge:
                break
            else:

                arr_indexes.append(old_index)
                old_rouge = rouge
                sentences_selected.append(all_sents_steamed[old_index])
                old_index = -1

        for i in range(len(all_sentences)):
            if i in arr_indexes:
                sentences_label.append('1' + ' ' + all_sentences[i])
            else:
                sentences_label.append('0' + ' ' + all_sentences[i])

        print(rouge)

        return rouge, '\n'.join(sentences_label)

    def convert_test_validate(self, test_files, dir_documents, dir_summaries, path_out, option):

        # option : test or validation
        num_test_files = len(test_files)
        print('num test files -' + option, num_test_files)

        rouges_test = []
        for file in test_files:
            print(file)
            id = file.split('_')[1]
            path_file = dir_documents + '/' + file
            path_summari = dir_summaries + '/summari_' + id
            path_file_out = path_out + option + '/' + file
            rouge, doc_labels = convert_extract.convert_extract_with_indexes([path_file], [path_summari])
            rouges_test.append(rouge)
            text_utils_english.write_text(doc_labels, path_file_out)

        print('aver rouge ' + option, sum(rouges_test) / len((rouges_test)))


if __name__ == "__main__":

    convert_extract = ConvertExtract()
    path_root_data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data'
    DIR_PATH_OUT = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'


    #########  separate CNN documents and hightlights  ###############


    # text_utils_english.separate_CNN_Data(path_data + "/dm_stories_tokenized",
    #                              path_data + '/dm')

    # text_utils_english.separate_CNN_Data(path_root_data + "/cnn_stories_tokenized",
    #                                       DIR_PATH_OUT)



    #########  labeling CNN documents with Greeding solution  ##########

    dir_doc_train = DIR_PATH_OUT + '/documents'
    dir_sum_train = DIR_PATH_OUT + '/summaries'
    dir_doc_test = DIR_PATH_OUT + '/test/doc_test'
    dir_sum_test = DIR_PATH_OUT + '/test/sum_test'
    dir_doc_valid = DIR_PATH_OUT + '/valid/doc_valid'
    dir_sum_valid = DIR_PATH_OUT + '/valid/sum_valid'

    path_out = DIR_PATH_OUT + '/data_labels/'

    ### convert train data ###

    list_docs_labels = []
    train_files = os.listdir(dir_doc_train)
    test_files = os.listdir(dir_doc_test)
    valid_files = os.listdir(dir_doc_valid)
    #
    k = 0
    rouges_train = []
    num_train_files = len(train_files)
    print('num train files', num_train_files)
    for i in range(num_train_files):
        file = train_files[i]
        id = file.split('_')[1]
        path_file = dir_doc_train + '/' + file
        path_summari = dir_sum_train + '/summari_' + id
        rouge, doc_labels = convert_extract.convert_extract_with_indexes([path_file], [path_summari])
        rouges_train.append(rouge)
        list_docs_labels.append(doc_labels)
        text_utils_english.write_text(doc_labels, path_out + 'train/' + file)

        if (i+1) % 1000 == 0 or i == num_train_files - 1:
            path_chunk_file = path_out + 'train_chunk/train_' + str(k)
            text_utils_english.write_text('\n####\n'.join(list_docs_labels), path_chunk_file)
            list_docs_labels = []
            k += 1

    print('aver rouge train', sum(rouges_train) / len((rouges_train)))


    ### convert validate and test data ###

    convert_extract.convert_test_validate(test_files, dir_doc_test, dir_sum_test, path_out, 'test')
    convert_extract.convert_test_validate(valid_files, dir_doc_valid, dir_sum_valid, path_out, 'valid')
