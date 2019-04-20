import shutil
import os
from sklearn.model_selection import train_test_split
from definitions import ROOT_DIR

root_data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'

def split_data_for_lead():
    path_data = root_data + '/documents'
    path_sums = root_data + '/summaries'
    path_data_sub = root_data + '/documents_for_lead'
    path_sums_sub = root_data + '/summaries_for_lead'
    listfiles_1  = open(ROOT_DIR + '/Models/Non_Colab/Centroid_W2V/list_cnn_filter_for_lead3').read().strip().split('\n')

    print(len(listfiles_1))

    for file in listfiles_1:
        id = file.split('_')[1]
        shutil.copy(path_data + '/' + file, path_data_sub + '/' + file)
        shutil.copy(path_sums + '/summari_' + id, path_sums_sub + '/summari_' + id)


if __name__ == '__main__':


    #split_data_for_lead()

    # split train, test, validate

    # 196,961/12,148/10,397
    # (90,160/1,216/1,090

    path_data = root_data + '/documents'
    path_sums = root_data + '/summaries'
    path_data_train = root_data + '/train/doc_train'
    path_sums_train = root_data + '/train/sum_train'
    path_data_test = root_data + '/test/doc_test'
    path_sums_test = root_data + '/test/sum_test'
    path_data_valid = root_data + '/valid/doc_valid'
    path_sums_valid = root_data + '/valid/sum_valid'

    list_all_files = os.listdir(path_data)

    #test_files = os.listdir(path_data_test)
    #train_and_validate_files = [file for file in list_all_files if file not in test_files]

    train_and_validate_files, test_files = train_test_split(list_all_files, test_size=0.01178, random_state=42)
    train_files, validate_files = train_test_split(train_and_validate_files, test_size=0.0133, random_state=42)

    # train_and_validate_files, test_files = train_test_split(list_all_files, test_size=0.047365, random_state=42)
    # train_files, validate_files = train_test_split(train_and_validate_files, test_size=0.05809, random_state=42)

    print(len(train_files))
    print(len(test_files))
    print(len(validate_files))


    # count = 7
    # for i in range(3):
    #
    #     path_test = path_data_test + '/' + str(count)
    #     os.mkdir(path_test)
    #     os.mkdir(path_test + '/doc_test')
    #     os.mkdir(path_test + '/sum_test')
    #
    #     train_and_validate_files, test_files = train_test_split(list_all_files, test_size=0.01178)
    #     for name_file in test_files:
    #         id = name_file.split('_')[1]
    #
    #         shutil.copy(path_data + '/' + name_file,path_test + '/doc_test/' + name_file)
    #         shutil.copy(path_sums + '/summari_' + id, path_test + '/sum_test/' + '/summari_' + id)
    #     count += 1

    for name_file in test_files:
        id = name_file.split('_')[1]

        shutil.copy(path_data + '/' + name_file, path_data_test + '/' + name_file)
        shutil.copy(path_sums + '/summari_' + id, path_sums_test + '/summari_' + id)

    for name_file in validate_files:
        id = name_file.split('_')[1]

        shutil.copy(path_data + '/' + name_file, path_data_valid + '/' + name_file)
        shutil.copy(path_sums + '/summari_' + id, path_sums_valid + '/summari_' + id)

    for name_file in train_files:
        id = name_file.split('_')[1]

        shutil.copy(path_data + '/' + name_file, path_data_train + '/' + name_file)
        shutil.copy(path_sums + '/summari_' + id, path_sums_train + '/summari_' + id)