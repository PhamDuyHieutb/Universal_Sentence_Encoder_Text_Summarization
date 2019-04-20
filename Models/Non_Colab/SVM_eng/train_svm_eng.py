from sklearn.svm import SVC
from joblib import dump
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from definitions import ROOT_DIR
from Models.SVM_eng import text_utils_eng

# X_train, Y_train, X_test, Y_test = text_utils.convert_features_svm('/home/hieupd/PycharmProjects/multi_summari_svm_vietnamese/svm_features')


PATH_CNN_FEATURES = ROOT_DIR + '/Data_Progress/cnn_features'
DATA_LABELS = ROOT_DIR + '/Data/CNN/data_labels_clus'



def read_text_file(pathfile):
    data = open(pathfile, 'r').read()

    return data


def get_labels_file(pathfile):

    f = open(pathfile).read().strip().split('\n')
    labels = [int(x[0]) for x in f]

    return labels


def get_labels(file_path):
    f = open(file_path).read().strip().split('\n')
    labels = [int(x[0]) for x in f]

    return labels


# def get_train_data():
#     train_files, test_files = prepare_train_test()
#     X_train = np.zeros(300)
#     Y_train = []
#     for file in train_files:
#         clus = file.split('.')[0]
#         X_train = np.vstack([X_train, np.load(PATH_CNN_FEATURES + '/' + file)])
#
#         Y_train += get_labels_clus(DATA_LABELS + '/' + clus)
#
#     X_test = np.zeros(300)
#     Y_test = []
#     for file in test_files:
#         clus = file.split('.')[0]
#         X_test = np.vstack([X_test, np.load(PATH_CNN_FEATURES + '/' + file)])
#
#         Y_test += get_labels_clus(DATA_LABELS + '/' + clus)
#
#     return X_train[1:], Y_train, X_test[1:], Y_test  # train_files, test_files

# get train or test data

def get_train_data():
    all_X_data = np.zeros(21)
    all_Y_data = []
    for chunk in os.listdir(ROOT_DIR + '/Data/CNN/data_labels_clus/train'):
        if chunk == 'train_27':
            id = chunk.split('_')[1]
            X_data = np.load(PATH_CNN_FEATURES + '/train/sentences_' + id + '.npy')
            Y_data = list(np.load(PATH_CNN_FEATURES + '/train/sentences_y_' + id + '.npy'))
            all_Y_data += Y_data
            print('get position ...')
            X_train_pos = text_utils_eng.get_position_train_features(X_data, Y_data, id)
            all_X_data = np.vstack((all_X_data, X_train_pos))

    return all_X_data[1:], np.array(all_Y_data)  # remove zeros initial

def get_test_data():
    all_X_test_pos = np.zeros(21)
    Y_test = []
    path_cnn_fea_test = PATH_CNN_FEATURES + '/test_1k'
    list_test_files = open(ROOT_DIR + '/Labels_Convert/Filters/test_files.txt').read().strip().split('\n')
    for file in list_test_files:
        X_test = np.load(path_cnn_fea_test + '/' + file.replace('story', 'npy'))
        X_test_pos = text_utils_eng.get_position_features_by_file(X_test)
        all_X_test_pos = np.vstack((all_X_test_pos, X_test_pos))
        Y_test += get_labels_file(DATA_LABELS + '/test/' + file.replace('npy','story'))

    return all_X_test_pos[1:], Y_test

def train(X_train, Y_train):
    model = SVC(kernel='rbf', C=32, gamma=0.03125, probability=True)
    model.fit(X_train, Y_train)
    dump(model, open('svm_model', 'wb'))

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
    print("Accuracy: %.4f%%" % (accuracy * 100.0))
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

            print("The best param are %s with a score of %0.3f" % (grid.best_params_, grid.best_score_))


def main():

    X_train, Y_train = get_train_data()
    X_test, Y_test = get_test_data()
    X_train = np.array(X_train)
    X_test = np.array(X_test)      #text_utils.concate_features(X_test, path_svm_features,"test")

    print(np.shape(X_train))
    print(np.shape(Y_train))


    print('training ...')
    model = train(X_train, Y_train)

    print('evaluating ...')
    evaluate(model, X_test, Y_test)


if __name__ == '__main__':
    main()
