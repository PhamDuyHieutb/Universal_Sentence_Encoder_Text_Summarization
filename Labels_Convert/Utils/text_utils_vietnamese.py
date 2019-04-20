from pyvi import ViTokenizer
import nltk
import re

SPECICAL_CHARACTER = {'(', ')', '[', ']', ',', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


def text_process_vietnamese(sentences):
    new_sentences = []

    for item in sentences:
        tmp = re.sub('[<>@~:.;]', '', item)
        tmp = re.sub('-', ' ', tmp)
        tmp = re.sub('[“”]', '"', tmp)
        text_tmp = []
        token_sent = ViTokenizer.tokenize(tmp).lower()

        for word in token_sent.split(' '):
            if len(word) != 1 or word in SPECICAL_CHARACTER:
                text_tmp.append(word)

        if len(text_tmp) > 5:
            new_sentences.append(' '.join(text_tmp).strip())

    return new_sentences


def triple_dot_process(sentences):
    sentences = [i for i in sentences if i != '']
    sents_processed = []
    for item in sentences:
        if "…" in item:
            b = item.split("…")
            if b[0] != '' and len(b[1]) > 20:
                if b[1][0].isupper() or b[1][1].isupper():
                    for i in b:
                        sents_processed.append(i)
                else:
                    sents_processed.append(item)
            else:
                sents_processed.append(item)

        if '...' in item:
            b = item.split('...')
            if b[0] != '' and len(b[1]) > 20:
                if b[1][0].isupper() or b[1][1].isupper():  # if second part has uppercase starting
                    for i in b:
                        sents_processed.append(i)
                else:
                    sents_processed.append(item)
            else:
                sents_processed.append(item)

        else:
            sents_processed.append(item)

    return sents_processed


def dot_process(sentences):
    sentences = [i for i in sentences if i != '']
    sents_dot_process = []
    for i in sentences:
        if i[-1] == '.':
            i = i[:-1]
        i_re = re.sub('[.]{2,}', ' ', i)  # remove dot at the end and triple dot
        if '.' in i_re:
            spl = i_re.split('.')
            if spl[0] != '' and len(spl[1]) > 20:
                if spl[1][0].isupper():  # if the next character is a Uppercase => split
                    for s in spl:
                        sents_dot_process.append(s)
                else:
                    sents_dot_process.append(i)
            else:
                sents_dot_process.append(i)
        else:
            sents_dot_process.append(i)

    return sents_dot_process


def split_sentences(file_name):
    # try:
    with open(file_name, 'r') as file:
        text_system = file.read().strip()

    text_system = [i.strip() for i in text_system.split('\n')]
    text_system = [i for i in text_system if i != '']       # remove space lines
    text_system_update = []  # update punctual


    for t in text_system:
        t = t.strip()
        if t[-1] != '.':
            text_system_update.append(t + '.')
        else:
            text_system_update.append(t)

    text_system_update = ' '.join(text_system_update)

    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    tmp = sentence_token.tokenize(text_system_update)

    tmp = triple_dot_process(tmp)

    sents_dot_process = dot_process(tmp)

    preprocess_sents = text_process_vietnamese(sents_dot_process)

    return preprocess_sents

def get_all_sentences(file_system, file_reference):
    sentences_origin_system = []
    for item in file_system:
        sentences_origin_system.append((item, split_sentences(item)))

    sentences_reference = []
    for item in file_reference:
        with open(item, 'r') as file:
            sentences_ref = text_process_vietnamese(nltk.sent_tokenize(file.read()))
            sentences_reference.append('. '.join(sentences_ref))

    return sentences_origin_system, sentences_reference