from consts import Consts
from features import Features
from chuliuwrapper import ChuLiuWrapper
from chu_liu import Digraph

import numpy as np
from functools import partial
import pickle
from time import time


class Training:
    def __init__(self, model: str, N: int, file_full_name: str, used_features: list):
        self.model = model
        self.N = N
        self.feature = Features(Consts.TRAIN, model, N, file_full_name, used_features)
        self.successors_per_sentence = ChuLiuWrapper(self.feature.hm_data).sentences_klicks
        self._load_w_parameter_from_last_iteration()
        self._perceptron()

        # Saves values
        with open("../data_from_training/" + self.model + "/" + str(N) + "/w_parameter", 'wb') as f:
            pickle.dump(self.w_parameter, f, protocol=-1)
        with open("../data_from_training/" + self.model + "/" + str(N) + "/w_as_list", 'w+') as f:
            for feature_weight in self.w_parameter:
                print(feature_weight, file=f)

    def get_score(self, sen_idx: int, h: int, m: int):
        # print("sen_idx:", sen_idx, ", h:", h, ", m:", m)
        # print("features_idxs:", self.feature.sentence_hm[(sen_idx, (h, m))])
        return np.sum(self.w_parameter[self.feature.sentence_hm[(sen_idx, (h, m))]])

    def _perceptron(self):
        Consts.print_info("perceptron", "Calculating")
        Consts.TIME = 1
        for n in range(self.N):
            t1 = time()
            for sen_idx in range(self.feature.sentences_amount):
                # t2 = time()
                y_max_successors = Digraph(self.successors_per_sentence[sen_idx],
                                           partial(self.get_score, sen_idx)).mst()
                y_org = self.feature.hm_match_feature[sen_idx]['f_score']
                y_max = self.get_f_score(sen_idx, y_max_successors.successors)
                self.w_parameter = self.w_parameter + y_org - y_max
                # Consts.print_time("perceptron per sentence: " + str(sen_idx), time() - t2)
            Consts.print_time("perceptron per iterate: " + str(n), time() - t1)

    def get_f_score(self, sen_idx, y: dict):
        f_score = np.zeros(self.feature.features_amount)
        for head in y:
            for modifier in y[head]:
                f_score[self.feature.sentence_hm[(sen_idx, (head, modifier))]] += 1

        return f_score

    def _load_w_parameter_from_last_iteration(self):
        if self.N == 20:
            self.w_parameter = np.zeros(self.feature.features_amount, dtype='int64')
        elif self.N == 50:
            with open("../data_from_training/" + self.model + "/20/w_parameter", 'rb') as f:
                self.w_parameter = pickle.load(f)
                self.N -= 20
        elif self.N == 80:
            with open("../data_from_training/" + self.model + "/50/w_parameter", 'rb') as f:
                self.w_parameter = pickle.load(f)
                self.N -= 50
        elif self.N == 100:
            with open("../data_from_training/" + self.model + "/80/w_parameter", 'rb') as f:
                self.w_parameter = pickle.load(f)
                self.N -= 80
