from __future__ import print_function, division
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import itertools
from definitions import ROOT_DIR
from sklearn.model_selection import train_test_split


# from Models.SVM_vietnamese.text_utils import write_file_text


def write_file_text(data, path_file):
    out = open(path_file, 'w')
    out.write(data)
    out.close()


# def get_position_features():
#     rpath = ROOT_DIR + '/Data/cnn_sub/cnn_features/'
#     output = ROOT_DIR + '/'
#     for file in os.listdir(rpath):
#         fea = np.load(rpath + '/' + file)
#         num_sents = len(fea)
#         posi_feas = np.array([1/(i+1) for i in range(num_sents)])
#         fea = np.concatenate((posi_feas, fea),axis=1)


def load_data_and_labels():
    x_text, y = load_sentences_and_labels()
    x_text = [s.split(" ") for s in x_text]  # split words in each sentence
    y = [[0, 1] if label == 1 else [1, 0] for label in y]
    return [x_text, y]


def load_sentences_and_labels():
    path_dir_data = ROOT_DIR + "/Data/CNN/data_labels_clus"  # "/content/data_labels"
    list_clus = os.listdir(path_dir_data)

    list_all_files = []
    for clus in list_clus:
        path_clus = path_dir_data + '/' + clus
        files = os.listdir(path_clus)
        list_all_files += [path_clus + '/' + file for file in files]

    # get list files train, test
    list_train_files, list_test_files = train_test_split(list_all_files, test_size=0.0118, shuffle=True)
    write_file_text('\n'.join(list_test_files), "test_files.txt")
    write_file_text('\n'.join(list_train_files), "train_files.txt")

    all_sents = []
    for l in [list_train_files, list_test_files]:
        for file in l:
            path_file = path_dir_data + '/' + file
            sents = open(path_file, 'r').read().strip().split('\n')
            all_sents += [s for s in sents if s != '']
        print('num sents', len(all_sents))
    labels = [int(x[0]) for x in all_sents]
    x_text = [a[2:] for a in all_sents]

    return x_text, labels


# load by file name to save embedding sentences for each filename
def load_sentences_and_labels_by_filename():
    all_sents = []
    path_dir_data = ROOT_DIR + "/Data/cnn_sub/data_labels"  # "/content/data_labels"

    list_train_files = open("train_files.txt").read().split('\n')
    list_test_files = open("test_files.txt").read().split('\n')

    for l in [list_train_files, list_test_files]:
        for file in l:
            sents = open(path_dir_data + '/' + file, 'r').read().strip().split('\n')
            sents = [s for s in sents if s != '']
            all_sents.append((file, len(sents), [int(x[0]) for x in sents]))
    return all_sents


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    transform word to index in dict of words
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    y = y.argmax(axis=1)

    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()

    sentences_padded = pad_sentences(sentences)
    # vocabulary is a dict of word
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


# for obtaining reproducible results
np.random.seed(0)
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
print('use_cuda = {}\n'.format(use_cuda))

mode = "nonstatic"
mode = "static"
# use_pretrained_embeddings = False
use_pretrained_embeddings = True

print('MODE      = {}'.format(mode))
print('EMBEDDING = {}\n'.format("pretrained" if use_pretrained_embeddings else "random"))

X, Y, vocabulary, vocabulary_inv_list = load_data()

print(np.shape(X))

vocab_size = len(vocabulary_inv_list)
sentence_len = X.shape[1]
num_classes = int(max(Y)) + 1  # added int() to convert np.int64 to int

print('vocab size       = {}'.format(vocab_size))
print('max sentence len = {}'.format(sentence_len))
print('num of classes   = {}'.format(num_classes))

ConvMethod = "in_channel__is_embedding_dim"
# ConvMethod = "in_channel__is_1"
embedding_dim = 300
num_filters = 10
kernel_sizes = [3, 4]
batch_size = 50


class CNN(nn.Module):
    def __init__(self, kernel_sizes=kernel_sizes, num_filters=num_filters, embedding_dim=embedding_dim,
                 pretrained_embeddings=None):
        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = mode == "nonstatic"

        if use_cuda:
            self.embedding = self.embedding.cuda()

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size + 1

            if ConvMethod == "in_channel__is_embedding_dim":
                conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size,
                                   stride=1)
            else:
                conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size * embedding_dim,
                                   stride=embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            )
            if use_cuda:
                component = component.cuda()

            conv_blocks.append(component)

            if 0:
                conv_blocks.append(
                    nn.Sequential(
                        conv1d,
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=maxpool_kernel_size)
                    ).cuda()
                )
        # moduleList : ghep cac block voi nhau de for qua cac block
        self.conv_blocks = nn.ModuleList(conv_blocks)  # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)  # full connected

    def forward(self, x):  # x: (batch, sentence_len)
        x = self.embedding(x)  # embedded x: (batch, sentence_len, embedding_dim)

        if ConvMethod == "in_channel__is_embedding_dim":
            #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.transpose(1, 2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        else:
            #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
            #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
            x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)

        x_list = [conv_block(x) for conv_block in self.conv_blocks]  # num dimension of vector
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        return F.softmax(self.fc(out), dim=1), feature_extracted


def create_sent_embedding(model, X, Y):
    X = torch.from_numpy(X).long()
    Y = torch.from_numpy(Y).long()

    dataset_X = TensorDataset(X, Y)
    X_loader = DataLoader(dataset_X, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    all_vectors = []
    predict_vecs = []

    for i, (inputs, labels) in enumerate(X_loader):
        inputs = Variable(inputs)
        preds, vectors = model(inputs)
        # preds: vector dung de predict
        # vectors: vectors dung de bieu dien

        all_vectors += list(vectors.cpu().data.numpy())
        predict_vecs += list(preds.cpu().data.numpy())

    return all_vectors, predict_vecs


def evaluate(model, x_test, y_test):
    dataset_test = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

    eval_acc = 0

    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        preds, vector = model(inputs)
        preds = torch.max(preds, 1)[1]  # lay max cua preds

        if use_cuda:
            preds = preds.cuda()
        # vectors.append(vector)
        eval_acc += (preds.data == labels.data).sum().item()

    acc = eval_acc / len(y_test)

    return acc  # , vector.cpu().data.numpy()


def load_pretrained_embeddings():
    path = ROOT_DIR + '/Models/Word2vec/models_english/cnn_extracted-python3.pl'  # '/content/gdrive/My Drive/20182_DOAN/2k_baomoi_extracted-python3.pl'
    #

    with open(path, 'rb') as f:
        embedding_weights = pickle.load(f)

    # embedding_weights is a dictionary {word_index:numpy_array_of_300_dim}
    out = np.array(
        list(embedding_weights.values()))  # added list() to convert dict_values to a list for use in python 3
    # np.random.shuffle(out)

    print('embedding_weights shape:', out.shape)
    # pretrained embeddings is a numpy matrix of shape (num_embeddings, embedding_dim)
    return out


pretrained_embeddings = load_pretrained_embeddings()


def train_test_one_split(cv, train_index, test_index):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    dataset_train = TensorDataset(x_train, y_train)
    # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).long()

    model = CNN(kernel_sizes, num_filters, embedding_dim, pretrained_embeddings)
    if cv == 0:
        print("\n{}\n".format(str(model)))

    if use_cuda:
        model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0001)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        tic = time.time()
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            preds, _ = model(inputs)
            if use_cuda:
                preds = preds.cuda()
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()  # backpropagate to  computes the gradients for all trainable parameters
            optimizer.step()

            # if 0:  # this does not improve the performance (even worse) (it was used in Kim's original paper)
            #     constrained_norm = 1  # 3 original parameter
            #     if model.fc.weight.norm().data[0] > constrained_norm:
            #         model.fc.weight.data = model.fc.weight.data * constrained_norm / model.fc.weight.data.norm()

        model.eval()

        eval_acc = evaluate(model, x_test, y_test)

        print('[epoch: {:d}] train_loss: {:.3f}   acc: {:.4f}   ({:.1f}s)'.format(1, loss.item(), eval_acc,
                                                                                  time.time() - tic))  # pytorch 0.4 and later
    vectors, predict_vectors = create_sent_embedding(model, X, Y)
    torch.save(model.state_dict(), "cnn_500.pth")  # '/content/gdrive/My Drive/20182_DOAN/cnn_model_2k'

    return vectors, predict_vectors


def do_cnn():
    length = len(X)
    train_index = np.array(range(0, 12677))
    test_index = np.array(range(12677, length))

    sentence_vecs, predict_vecs = train_test_one_split(0, train_index, test_index)
    print('cv = {}    train size = {}    test size = {}\n'.format(0, len(train_index), len(test_index)))

    # print('\navg acc = {:.3f}   (total time: {:.1f}s)\n'.format(sum(acc_list) / len(acc_list), time.time() - tic))
    # save extracted sentence vectors in case that we can reuse it for other purpose (e.g. used as input to an SVM classifier)
    # each vector can be used as a fixed-length dense vector representation of a sentence

    np.save('sentences.npy', np.array(sentence_vecs))
    np.save('sentences_y.npy', Y)

    # sentence_vecs = np.load("/home/hieupd/PycharmProjects/CNN_SVM_summarization/Models/cnn_colab/sentences.npy")
    # save sentence embedding in each file perspectively

    pathout = ROOT_DIR + '/Data/cnn_sub/cnn_features/'  # 'content/gdrive/My Drive/20182_DOAN/cnn_features/'  #
    all_sents = load_sentences_and_labels_by_filename()
    pivot = 0
    for sents in all_sents:
        filename = sents[0]
        num_sents = sents[1]
        labels_real = sents[2]
        vectors = sentence_vecs[pivot:(pivot + num_sents)]
        labels = Y[pivot:(pivot + num_sents)]
        pivot += num_sents
        if np.array_equal(labels, labels_real):
            np.save(pathout + sents[0], vectors)
        else:
            print(filename)
            print(labels_real)
            print(labels)


def main():
    do_cnn()


if __name__ == "__main__":
    main()
