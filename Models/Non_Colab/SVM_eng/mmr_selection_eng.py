from Models.Non_Colab.SVM_eng import text_utils_eng
import math


def get_idf(all_idf, word):
    if word in all_idf:

        return all_idf[word]
    else:

        return 0.1


def sentence_sim(sentence1, sentence2, idf):
    numerator = 0
    denom1 = 0
    denom2 = 0
    try:

        sentence1 = sentence1[2]
        dict_w_1 = text_utils_eng.get_freq_words_from_doc(sentence1)

        sentence2 = sentence2[2]
        dict_w_2 = text_utils_eng.get_freq_words_from_doc(sentence2)

        for word in sentence2.split(' '):
            tf_word_2 = text_utils_eng.get_word_freq(word, dict_w_2)
            tf_word_1 = text_utils_eng.get_word_freq(word, dict_w_1)
            idf_w = get_idf(idf, word)
            numerator += tf_word_1 * tf_word_2 * (idf_w ** 2)
            denom2 += (tf_word_2 * idf_w) ** 2

        for word in sentence1.split(" "):
            tf_word_1 = text_utils_eng.get_word_freq(word, dict_w_1)
            idf_w = get_idf(idf, word)

            denom1 += (tf_word_1 * idf_w) ** 2
        return numerator / (math.sqrt(denom2) * math.sqrt(denom1))
    except ZeroDivisionError:
        return float('-inf')


# Si,j = [{"doc": index, "sent": sent}, ...]
def MMRScore(Si, summari, lambta, idf):
    l_expr = lambta * Si[3] * Si[4]  # score and position feature
    value = [float("-inf")]

    for sent in summari:
        # for each sent in summari, we compute similarity with new sent
        Sim2 = sentence_sim(Si, sent, idf)
        value.append(Sim2)

    r_expr = (1 - lambta) * max(value)
    MMR_SCORE = l_expr - r_expr

    return MMR_SCORE


def make_summary(sentences, lambta):
    sent_cleaned_stemed = [sent[2] for sent in sentences]
    idf = text_utils_eng.get_idf_sklearn(sent_cleaned_stemed)

    summary = [sentences[0]]
    del sentences[0]

    while len(summary) < 3:

        index = -1
        max_score_mmr = -1
        for i in range(len(sentences)):
            sent = sentences[i]
            mmr_score = MMRScore(sent, summary, lambta, idf)
            if mmr_score > max_score_mmr:
                max_score_mmr = mmr_score
                index = i

        sent_selected = sentences[index]  # add sentence which has MMRScore max
        summary.append(sent_selected)

        del sentences[index]

        if len(sentences) == 0:
            break


    summary = [sent[1] for sent in summary]

    return ' . '.join(summary)
