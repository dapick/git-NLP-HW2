from chuliuwrapper import ChuLiuWrapper
from parsing import Parsing
from consts import Consts
from model import BasicModel
from chu_liu import Digraph

from multiprocessing.pool import Pool
from time import time
import numpy as np
from functools import partial
import pickle


class Inference(object):
    def __init__(self, file_full_name: str, model: str, N: int):
        self.file_full_name = file_full_name
        self.model_type = model
        self.labeled_file_name = self._get_labeled_file_name()

        if model == Consts.BASIC_MODEL:
            self.model = BasicModel(Consts.LABEL, N, file_full_name)
        # elif model == Consts.ADVANCED_MODEL:
        #     self.model = AdvancedModel(Consts.TAG)

        with open("../data_from_training/" + self.model_type + "/" + str(N) + "/w_parameter", 'rb') as f:
            self.model.w_parameter = pickle.load(f)

        self.successors_per_sentence = ChuLiuWrapper(self.model.feature.hm_data).sentences_klicks

    def _get_labeled_file_name(self):
        ret_file = self.file_full_name
        file_name = ret_file.split('.unlabeled')
        if self.model_type == Consts.BASIC_MODEL:
            ret_file = file_name[0] + "_m1_"
        elif self.model_type == Consts.ADVANCED_MODEL:
            ret_file = file_name[0] + "_m2_"
        ret_file = ret_file + "302988217.labeled"
        return ret_file

    def get_score(self, sen_idx: int, h: int, m: int):
        return np.sum(self.model.w_parameter[self.model.feature.sentence_hm[(sen_idx, (h, m))]])

    def _label_sentence(self, sen_idx):
        Consts.TIME = 1
        t1 = time()
        y_max = Digraph(self.successors_per_sentence[sen_idx], partial(self.get_score, sen_idx)).mst().successors
        heads = np.zeros(len(self.model.feature.hm_data[sen_idx]['words']), dtype='int64')
        for head in y_max:
            for modifier in y_max[head]:
                heads[modifier] = head

        Consts.print_time("Labeling sentence " + str(sen_idx + 1), time() - t1)
        return heads

    def label(self):
        Consts.print_info("label", "Labeling file '" + self.file_full_name + "' by '" + self.model_type + "'")
        Consts.TIME = 1
        t1 = time()

        # Run parallel - good when checking many sentences
        with Pool(3) as pool:
            labels_list_per_sentence = pool.map(self._label_sentence, range(self.model.feature.sentences_amount))

        Parsing().parse_list_of_dict_to_labeled_file(self.labeled_file_name, self.model.feature.hm_data,
                                                     list(labels_list_per_sentence))
        Consts.print_time("Labeling file", time() - t1)

    def calculate_accuracy(self, out_file: str, expected_file: str):
        out_dicts = Parsing().parse_labeled_file_to_list_of_dict(out_file)
        exp_dicts = Parsing().parse_labeled_file_to_list_of_dict(expected_file)

        count = 0
        count_eq = 0
        for sen_idx, sentence in enumerate(out_dicts):
            for head_idx, head in enumerate(sentence['heads']):
                count += 1
                if sentence['heads'][head_idx] == exp_dicts[sen_idx]['heads'][head_idx]:
                    count_eq += 1

        print(str(count_eq * 100 / count) + "%")
