from parsing import Parsing
from consts import Consts

import pickle
from time import time
import numpy as np


class Features:
    def __init__(self, method: str, model: str, N: int, file_full_name: str = None, used_features: list = None):
        self.model = model
        self.N = N

        if method == Consts.TRAIN:
            self._training(used_features, file_full_name)

        elif method == Consts.LABEL:
            self._set_internal_values(file_full_name)

    def _training(self, used_features: list, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.used_features = used_features
        self.idx = 0
        self.feature_vector = {}
        self.hm_match_feature = []
        self.features_occurrences = []

        self.hm_data = Parsing().parse_labeled_file_to_list_of_dict(file_full_name)

        self._light_features()
        self.features_amount = len(self.feature_vector)
        self.sentences_amount = len(self.hm_data)

        self._calculate_features_for_all_sen_hm()
        self._calculate_f_score_per_sentence()

        # Saves values
        with open("../data_from_training/" + self.model + "/" + str(self.N) + "/internal_values_of_feature", 'wb') as f:
            pickle.dump([self.feature_vector, self.used_features], f, protocol=-1)
        with open("../data_from_training/" + self.model + "/" + str(self.N) + '/feature_vector', 'w+') as f:
            for key, values in self.feature_vector.items():
                f.write(str(key) + " => " + str(values) + "\n")

    def _set_internal_values(self, file_full_name: str):
        # Restore values
        with open("../data_from_training/" + self.model + "/" + str(self.N) + "/internal_values_of_feature", 'rb') as f:
            self.feature_vector, self.used_features = pickle.load(f)

        self.hm_data = Parsing().parse_unlabeled_file_to_list_of_dict(file_full_name)
        self._calculate_features_for_all_sen_hm()
        self.sentences_amount = len(self.hm_data)

    def _light_features(self):
        for _ in range(0, len(self.hm_data)):
            self.hm_match_feature.append({})

        for feature in self.used_features:
            Consts.print_info("_light_features", "Building feature " + feature)
            for sen_idx, sentence in enumerate(self.hm_data):
                for h_m in sentence['edges']:
                    tup = self._feature_hm_match_data(sen_idx, h_m, feature)
                    self.feature_structure((feature, tup))
                    if h_m not in self.hm_match_feature[sen_idx]:
                        self.hm_match_feature[sen_idx][h_m] = []
                    self.hm_match_feature[sen_idx][h_m].append(self.feature_vector[(feature, tup)][0])

    # Gives an index for each feature and count how many times it was used
    def feature_structure(self, keys: tuple):
        if keys not in self.feature_vector:
            self.feature_vector[keys] = [self.idx, 1]
            self.features_occurrences.append(1)
            self.idx += 1
        else:
            self.feature_vector[keys][1] += 1
            self.features_occurrences[self.feature_vector[keys][0]] += 1

    # given feature and (h,m) returns relevant data by feature number
    def _feature_hm_match_data(self, sen_idx: int, hm: tuple, feature: str):
        p = hm[0]
        c = hm[1]
        return {
            "1": (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['tags'][p]),
            "2": (self.hm_data[sen_idx]['words'][p]),
            "3": (self.hm_data[sen_idx]['tags'][p]),
            "4": (self.hm_data[sen_idx]['words'][c], self.hm_data[sen_idx]['tags'][c]),
            "5": (self.hm_data[sen_idx]['words'][c]),
            "6": (self.hm_data[sen_idx]['tags'][c]),
            "7": (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['tags'][p],
                  self.hm_data[sen_idx]['words'][c], self.hm_data[sen_idx]['tags'][c]),
            "8": (self.hm_data[sen_idx]['tags'][p], self.hm_data[sen_idx]['words'][c],
                  self.hm_data[sen_idx]['tags'][c]),
            "9": (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['words'][c],
                  self.hm_data[sen_idx]['tags'][c]),
            "10": (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['tags'][p],
                   self.hm_data[sen_idx]['tags'][c]),
            "11": (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['tags'][p],
                   self.hm_data[sen_idx]['words'][c]),
            "12": (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['words'][c]),
            "13": (self.hm_data[sen_idx]['tags'][p], self.hm_data[sen_idx]['tags'][c])
        }[feature]

    def print_features_to_file(self, dir_name: str):
        with open(dir_name + self.model + '/feature_vector', 'w+') as f:
            for key, values in self.feature_vector.items():
                f.write(str(key) + " => " + str(values) + "\n")

        with open(dir_name + self.model + '/h_m_match_to_feature', 'w+') as f:
            for i, x in enumerate(self.hm_match_feature):
                f.write(str(i) + " =>\n")
                for key, val in x.items():
                    f.write("\t" + str(key) + " => " + str(val) + "\n")

    def get_features_idx_per_h_m(self, sen_idx, hm):
        features_idx = []
        for feature in self.used_features:
            key = self._feature_hm_match_data(sen_idx, hm, feature)
            if self.feature_vector.get((feature, key)):
                features_idx.append(self.feature_vector[(feature, key)][0])

        return features_idx

    # Saves for each sentence and (h, m) a list of the features its light
    def _calculate_features_for_all_sen_hm(self):
        Consts.print_info("_calculate_features_for_all_sen_hm", "Preprocessing")
        Consts.TIME = 1
        t1 = time()
        self.sentence_hm = {}

        for sen_idx, sentence in enumerate(self.hm_data):
            for i in range(len(sentence['words'])):
                for j in range(1, len(sentence['words'])):
                    if i == j:
                        continue
                    self.sentence_hm[(sen_idx, (i, j))] = np.array(self.get_features_idx_per_h_m(sen_idx, (i, j)))

        Consts.print_time("_calculate_features_for_all_sen_hm", time() - t1)

    def _calculate_f_score_per_sentence(self):
        Consts.print_info("_calculate_f_score_per_sentence", "Preprocessing")
        Consts.TIME = 1
        t1 = time()
        f_scores = []
        for sen_idx in range(self.sentences_amount):
            f_scores.append(np.zeros(self.features_amount))

        for sen_idx, sen_dict in enumerate(self.hm_match_feature):
            for features_list in sen_dict.values():
                f_scores[sen_idx][features_list] += 1

        for sen_idx, sen_dict in enumerate(self.hm_match_feature):
            sen_dict['f_score'] = f_scores[sen_idx]
        Consts.print_time("_calculate_f_score_per_sentence", time() - t1)
