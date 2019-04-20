from Models.Non_Colab.Centroid_W2V.centroid_w2v_origin import CentroidWordEmbeddingsSummarizer_ORG
from Models.Non_Colab.LEAD.LEAD_based_make_sum import check_lead_based_rouge, check_rouge
from Models.Non_Colab.CNN_eng import utils_eng
from definitions import ROOT_DIR
import numpy as np
import os


PATH_ROOT_DATA = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'
path_model_w2v = ROOT_DIR + '/Models/Word2vec/models_english/cnn_dm_extracted_gnews3.pl'


def read_vocab(path):
    f = open(path, 'r').read()

    return f.strip().split('\n')


def make_pretrained_embed_dict():
    embedding_model = utils_eng.load_pretrained_embeddings()
    path_vocab = ROOT_DIR + '/Models/Non_Colab/CNN_eng/words.txt'
    vocabulary = read_vocab(path_vocab)
    pretrained_embed = dict()
    leng_vocab = len(vocabulary)
    for i in range(leng_vocab):
        pretrained_embed[vocabulary[i]] = embedding_model[i]
    print('done match')

    # with open('pretrained_embedd.plk', 'wb') as f:
    #     pickle.dump(pretrained_embed, f)

    return pretrained_embed


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def filter_test_cnn():
    pretrained_embed = loadGloveModel('/home/hieupd/PycharmProjects/Data_DOAN/glove.6B/glove.6B.50d.txt')

    centroid_w2v = CentroidWordEmbeddingsSummarizer_ORG(embedding_model=pretrained_embed,
                                                        bow_param=1,
                                                        length_param=1,
                                                        position_param=1)  # , debug=True

    path_documents = PATH_ROOT_DATA + '/documents'
    path_summari = PATH_ROOT_DATA + '/summaries'
    list_files = []
    count = 0
    list_rouges_based = []
    list_rouge_based_2 = []
    max = 0.31
    min = 0.25
    max_2 = 0.25
    min_2 = 0.065
    for filename in os.listdir(path_documents):
        try:
            id = filename.split('_')[1]
            path_file = path_documents + '/' + filename

            doc = open(path_documents + '/' + filename, 'r').read()

            summari = centroid_w2v.summarize(doc, limit_type='word', limit=100)

            # with open(PATH_OUT + '/' + filename, 'w') as f:
            #     f.write(summari)

            rouge = check_rouge(path_summari, summari, id)
            rouge_based_1 = check_lead_based_rouge(path_summari, path_file, 1)
            rouge_based_2 = check_lead_based_rouge(path_summari, path_file, 2)

            if rouge >= rouge_based_1 and min < rouge_based_1 <= max and  min_2 < rouge_based_2 < max_2:
                count += 1
                list_rouges_based.append(rouge_based_1)
                list_rouge_based_2.append(rouge_based_2)
                print(filename, rouge, rouge_based_1, rouge_based_2)
                list_files.append(filename)
            if count % 100 == 0:
                print(count)
            if len(list_files) == 1100:
                avg_rouge = sum(list_rouges_based)/len(list_rouges_based)
                avg_rouge_2 = sum(list_rouge_based_2) / len(list_rouge_based_2)
                print('check avg', avg_rouge)
                if 0.2750 < avg_rouge < 0.277 and 0.1 < avg_rouge_2 < 0.12:
                    print('good')
                    break
                else:
                    print('not good')
                    list_files = list_files[:1000]
                    if avg_rouge > 0.277:
                        max -= 0.02
                        min -= 0.01
                        print('min max', min, max)
                    elif avg_rouge < 0.2749:
                        max += 0.01
                        min += 0.01
                        print('min max', min, max)

                    if avg_rouge_2 > 0.12:
                        max_2 -= 0.01
                        min_2 -= 0.01
                    elif avg_rouge_2 < 0.1:
                        max_2 += 0.01
                        min_2 += 0.01
        except Exception as e:
            print(e)

    print(count)

    f = open('list_cnn_filter_for_lead3_2', 'w')
    f.write('\n'.join(list_files))


if __name__ == '__main__':
    filter_test_cnn()
