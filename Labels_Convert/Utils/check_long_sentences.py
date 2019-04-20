import os
from definitions import ROOT_DIR



def check_long_sents_CNN_data():
    max= ('',0, '')
    path_dir_data = ROOT_DIR + "/Data/CNN/data_labels_clus"
    for clus in os.listdir(path_dir_data):
        print(clus)
        path_clus = path_dir_data + '/' + clus
        if clus != 'train':
            for filename in os.listdir(path_clus):
                sents = open(path_clus + '/' + filename, 'r').read().strip().split('\n')
                all_sents = [s[2:] for s in sents if s != '']
                for sen in all_sents:
                    leng = len(sen.split(' '))
                    if leng > max[1]:
                        max = (path_clus + '/' + filename, leng, sen)
        else:
            for filename in os.listdir(path_clus):
                all_sents = []
                docs = open(path_clus + '/' + filename, 'r').read().strip().split('\n####\n')
                for doc in docs:
                    all_sents += [s[2:] for s in doc.split('\n') if s != '']
                for sen in all_sents:
                    leng = len(sen.split(' '))
                    if leng > max[1]:
                        max = (path_clus + '/' + filename, leng, sen)


    print(max)

def check_sents_length():
    max = ('', 0, '')
    path_dir_data =  "/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn/documents"
    for file in os.listdir(path_dir_data):
        sents = open(path_dir_data + '/' + file, 'r').read().strip().split('. ')
        for s in sents:
            leng = len(s.split(' '))
            if leng > 200:
                print(path_dir_data + '/' + file, leng, s)
            if leng > max[1]:
                max = (path_dir_data + '/' + file, leng, s)

    print(max)

check_sents_length()