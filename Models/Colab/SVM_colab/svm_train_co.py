from sklearn.svm import SVC
from joblib import dump
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
from definitions import ROOT_DIR
from Models.Non_Colab.SVM_vietnamese import text_utils


def read_text_file(pathfile):
    data = open(pathfile, 'r').read()

    return data


def get_labels_clus(pathclus):
    labels = []
    for file in os.listdir(pathclus):
        f = open(pathclus + '/' + file).read().strip().split('\n')
        labels += [int(x[0]) for x in f]

    return labels


def get_labels(file_path):
    f = open(file_path).read().strip().split('\n')
    labels = [int(x[0]) for x in f]

    return labels


# get train or test data
def get_train_test_cnn_features(option):
    DATA_LABELS = "/home/hieupd/PycharmProjects/CNN_SVM_summarization/Data/cnn_sub/data_labels" # '/content/gdrive/My Drive/20182_DOAN/data_labels'
    root =  "/home/hieupd/PycharmProjects/CNN_SVM_summarization/Data/cnn_sub/cnn_features" # "/content/gdrive/My Drive/20182_DOAN/cnn_features/"

    if option == "train":
        path_list_files = ROOT_DIR + '/Models/CNN_eng/train_files.txt'
    else:
        path_list_files = ROOT_DIR + '/Models/CNN_eng/test_files.txt'

    list_files = open(path_list_files).read().split('\n')

    X_data = np.zeros(20)
    Y_data = []
    for filename in list_files:
        X_data = np.vstack([X_data, np.load(root + '/' + filename + '.npy')])

        Y_data += get_labels(DATA_LABELS + '/' + filename)

    return X_data[1:], Y_data

def get_train_test_svm_features(option):
    root =  "/home/hieupd/PycharmProjects/CNN_SVM_summarization/Data/cnn_sub/svm_features" # "/content/gdrive/My Drive/20182_DOAN/cnn_features/"

    if option == "train":
        path_list_files = ROOT_DIR + '/Models/CNN_eng/train_files.txt'
    else:
        path_list_files = ROOT_DIR + '/Models/CNN_eng/test_files.txt'

    list_files = open(path_list_files).read().split('\n')

    X_data = []
    Y_data = []
    for file in list_files:
        path_file = root + '/' + file
        features = open(path_file, 'r').read().split('\n')
        X_data += [fea[2:].strip().split(' ') for fea in features]
        Y_data +=  [int(fea[0]) for fea in features]

    return X_data, Y_data


def train(X_train, Y_train):
    model = SVC(kernel='rbf', C=10, gamma=1)  # probability=True
    model.fit(X_train, Y_train)
    dump(model, open('model_test', 'wb'))

    return model


def get_doc_id(index, leng_docs):
    '''

    :param index: index of sent in array all sentences
    :param leng_docs: is a dict has num sent of each doc
    :return:
    '''
    sum_len = 0
    id = 0
    for i in leng_docs:
        if index < sum_len + leng_docs[i]:
            id = i
            break
        else:
            sum_len += leng_docs[i]

    return id


def evaluate(model, X_test, Y_test):
    # Y_predict = model.predict_proba(X_test)
    Y_predict = model.predict(X_test)
    predictions = [round(value) for value in Y_predict]
    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # return Y_predict


def evaluate_prob_data(model, X_test):
    # predict document input

    # reload = load("model_test")
    predict = model.predict_proba(X_test)
    sort_predict = sorted(enumerate(predict), key=lambda x: list(x[1])[0], reverse=True)
    return sort_predict


def train_test(X_train, Y_train):
    for c in np.arange(1, 10, 2):
        c_end = c + 2
        C_range = np.logspace(c, c_end, 2)
        for g in np.arange(-5, 5, 2):
            g_end = g + 4
            gamma_range = np.logspace(g, g_end, 2)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X_train, Y_train)

            print("The best param are %s with a score of %0.4f" % (grid.best_params_, grid.best_score_))


def main():
    path_svm_features = "/home/hieupd/PycharmProjects/CNN_SVM_summarization/Data/cnn_sub/svm_features"

    print("getting data train")
    X_train, Y_train = get_train_test_cnn_features("train")
    print("getting data test")
    X_test, Y_test = get_train_test_cnn_features("test")
    X_train = text_utils.concate_features(X_train, path_svm_features, "train") # X_train = np.array(X_train)
    X_test =  text_utils.concate_features(X_test, path_svm_features,"test")  # X_test = np.array(X_test)  #

    print(np.shape(X_train))
    print(np.shape(Y_train))

    print("training ...")
    model = train(X_train, Y_train)

    print("evaluating ...")
    evaluate(model, X_test, Y_test)

    #train_test(X_train, Y_train)

if __name__ == '__main__':
    main()
