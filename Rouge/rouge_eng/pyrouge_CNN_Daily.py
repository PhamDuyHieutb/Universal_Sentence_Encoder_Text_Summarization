from pyrouge import Rouge155
import time
from definitions import ROOT_DIR

start_time = time.time()

if __name__ == "__main__":
    #dir_name = '/home/huong/api_summarization_multidoc/hieupd/'
    rouge_dir = ROOT_DIR + '/Rouge/rouge_eng/ROUGE-1.5.5'
    data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'

    # -n 2 -m -u -c 95 -r 1000 -f A -p 0.5 -t 0
    rouge_args = '-e ROUGE-1.5.5/data -n 2 -m -u -c 95 -f A -p 0.5 -t 0 -a'

    # '-e', self._rouge_data,                           # '-a',  # evaluate all systems
    # '-n', 4,  # max-ngram                             # '-x',  # do not calculate ROUGE-L
    # '-2', 4,  # max-gap-length                        # '-u',  # include unigram in skip-bigram
    # '-c', 95,  # confidence interval                  # '-r', 1000,  # number-of-samples (for resampling)
    # '-f', 'A',  # scoring formula                     # '-p', 0.5,  # 0 <= alpha <=1
    # '-t', 0,  # count by token instead of sentence    # '-d',  # print per evaluation scores

    rouge = Rouge155(rouge_dir, rouge_args)
    #rouge = Rouge155()

    rouge.model_dir = data + '/test/sum_test'
    rouge.model_filename_pattern = 'summari_#ID#'


    #rouge.system_dir = data + '/test/sum_lead3'
    rouge.system_dir = ROOT_DIR + '/Data_Progress/summaries/universal_3'
    rouge.system_filename_pattern = 'system_(\d+)'


    print("-------------------------------------------")

    rouge_output = rouge.convert_and_evaluate()
    print(rouge_output)

    print("Execution time: " + str(time.time() - start_time))
