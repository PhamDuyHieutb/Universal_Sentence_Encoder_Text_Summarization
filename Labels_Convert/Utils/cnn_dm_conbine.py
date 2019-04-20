import shutil
import os

# root_data_cnn = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn/data_labels'
# root_data_dm = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/dm/data_labels'
# root_output  = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn_dm'

root_data_cnn = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'
root_data_dm = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/dm'
root_output  = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn_dm'


# path_cnn_train = root_data_cnn + '/train'
# path_cnn_test = root_data_cnn + '/test'
# path_cnn_valid = root_data_cnn + '/valid'
#
# path_dm_train = root_data_dm + '/train'
# path_dm_test = root_data_dm + '/test'
# path_dm_valid = root_data_dm + '/valid'
#
# path_output_train = root_output + '/train'
# path_output_test = root_output + '/test'
# path_output_valid = root_output + '/valid'

id = 0
# for typ_data in ['train', 'test', 'valid']:
#     path_cnn = root_data_cnn + '/' + typ_data
#     path_output_data = root_output + '/' + typ_data
#     list_doc_cnn = os.listdir(path_cnn)
#     for file in list_doc_cnn:
#         id += 1
#         if id < 10:
#             shutil.copy(path_cnn + '/' + file, path_output_data + '/doc_0' + str(id))
#             #shutil.copy(path_summari_cnn + '/summari_' + x, path_output_sum + '/summari_0' + str(id))
#         else:
#             shutil.copy(path_cnn + '/' + file, path_output_data + '/doc_' + str(id))
#             #shutil.copy(path_summari_cnn + '/summari_' + x, path_output_sum + '/summari_' + str(id))
#         if id % 1000 == 0:
#             print(id)
#
#     path_dm = root_data_dm + '/' + typ_data
#
#     list_doc_dm = os.listdir(path_dm)
#     for file in list_doc_dm:
#         id += 1
#
#         shutil.copy(path_dm + '/' + file, path_output_data + '/doc_' + str(id))
#         #shutil.copy(path_summari_dm + '/summari_' + x, path_output_sum + '/summari_' + str(id))
#         if id % 1000 == 0:
#             print(id)

data_test = root_data_cnn + '/documents_test'
list_doc_cnn = os.listdir(data_test)
for file in list_doc_cnn:
    index = file.split('_')[1]
    id += 1
    if id < 10:
        shutil.copy(root_data_cnn + '/summaries_test/summari_' + index, root_output + '/summaries_test/summari_0' + str(id))
        shutil.copy(data_test + '/' + file, root_output + '/documents_test/doc_0' + str(id))

    else:
        shutil.copy(root_data_cnn + '/summaries_test/summari_' + index, root_output + '/summaries_test/summari_' + str(id))
        shutil.copy(root_data_cnn + '/documents_test/' + file, root_output + '/documents_test/doc_' + str(id))

    if id % 1000 == 0:
        print(id)

data_dm = root_data_dm + '/documents_test'
list_doc_dm = os.listdir(data_dm)
for file in list_doc_dm:
    index = file.split('_')[1]
    id += 1

    shutil.copy(root_data_dm + '/summaries_test/summari_' + index, root_output + '/summaries_test/summari_' + str(id))
    shutil.copy(root_data_dm + '/documents_test/' + file, root_output + '/documents_test/doc_' + str(id))

    if id % 1000 == 0:
        print(id)