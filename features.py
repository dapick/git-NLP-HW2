from parsing import Parsing
from consts import Consts

import pickle
from time import time
import numpy as np


class Features:
    def __init__(self, method: str, model: str, N: int, file_full_name: str = None, used_features: list = None):
        self.model = model
        self.N = N
        self.features_funcs = {"1": self.feature_1, "2": self.feature_2, "3": self.feature_3, "4": self.feature_4,
                               "5": self.feature_5, "6": self.feature_6, "7": self.feature_7, "8": self.feature_8,
                               "9": self.feature_9, "10": self.feature_10, "11": self.feature_11, "12": self.feature_12,
                               "13": self.feature_13, "tags_between": self.feature_tags_between,
                               "contextual_tags": self.feature_contextual_tags}

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
        with open("../data_from_training/" + self.model + "/" + str(self.N) + '/h_m_match_to_feature', 'w+') as f:
            for i, x in enumerate(self.hm_match_feature):
                f.write(str(i) + " =>\n")
                for key, val in x.items():
                    f.write("\t" + str(key) + " => " + str(val) + "\n")

    def _set_internal_values(self, file_full_name: str):
        # Restores values
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
                    keys_per_feature = self.features_funcs[feature](sen_idx, h_m)
                    for keys in keys_per_feature:
                        feature_idx = self.feature_structure(keys)
                        if h_m not in self.hm_match_feature[sen_idx]:
                            self.hm_match_feature[sen_idx][h_m] = []
                        self.hm_match_feature[sen_idx][h_m].append(feature_idx)

    # Gives an index for each feature and count how many times it was used
    def feature_structure(self, keys: tuple):
        if keys not in self.feature_vector:
            self.feature_vector[keys] = [self.idx, 1]
            self.features_occurrences.append(1)
            feature_idx = self.idx
            self.idx += 1
        else:
            self.feature_vector[keys][1] += 1
            self.features_occurrences[self.feature_vector[keys][0]] += 1
            feature_idx = self.feature_vector[keys][0]

        return feature_idx

    @staticmethod
    def get_direction_and_distance(parent, child):
        direction = 'R' if child > parent else 'L'
        distance = abs(parent - child)
        if 5 < distance < 10:
            distance = 5
        elif distance > 10:
            distance = 10
        return direction, distance

    def feature_1(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("1", (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['tags'][p]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            p_word = self.hm_data[sen_idx]['words'][p]
            p_tag = self.hm_data[sen_idx]['tags'][p]
            keys = [("1", (p_word, p_tag))]
            keys += [("1", (p_word, p_tag, direction, distance))]
            return keys

    def feature_2(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("2", (self.hm_data[sen_idx]['words'][p]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            p_word = self.hm_data[sen_idx]['words'][p]
            keys = [("2", p_word)]
            keys += [("2", (p_word, direction, distance))]
            return keys

    def feature_3(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("3", (self.hm_data[sen_idx]['tags'][p]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            p_tag = self.hm_data[sen_idx]['tags'][p]
            keys = [("3", p_tag)]
            keys += [("3", (p_tag, direction, distance))]
            return keys

    def feature_4(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("4", (self.hm_data[sen_idx]['words'][c], self.hm_data[sen_idx]['tags'][c]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            c_word = self.hm_data[sen_idx]['words'][c]
            c_tag = self.hm_data[sen_idx]['tags'][c]
            keys = [("4", (c_word, c_tag))]
            keys += [("4", (c_word, c_tag, direction, distance))]
            return keys

    def feature_5(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("5", (self.hm_data[sen_idx]['words'][c]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            c_word = self.hm_data[sen_idx]['words'][c]
            keys = [("5", c_word)]
            keys += [("5", (c_word, direction, distance))]
            return keys

    def feature_6(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("6", (self.hm_data[sen_idx]['tags'][c]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            c_tag = self.hm_data[sen_idx]['tags'][c]
            keys = [("6", c_tag)]
            keys += [("6", (c_tag, direction, distance))]
            return keys

    def feature_7(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        direction, distance = self.get_direction_and_distance(p, c)
        p_word = self.hm_data[sen_idx]['words'][p]
        p_tag = self.hm_data[sen_idx]['tags'][p]
        c_word = self.hm_data[sen_idx]['words'][c]
        c_tag = self.hm_data[sen_idx]['tags'][c]
        keys = [("7", (p_word, p_tag, c_word, c_tag))]
        keys += [("7", (p_word, p_tag,c_word, c_tag, direction, distance))]
        return keys

    def feature_8(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            p_tag = self.hm_data[sen_idx]['tags'][p]
            c_word = self.hm_data[sen_idx]['words'][c]
            c_tag = self.hm_data[sen_idx]['tags'][c]
            keys = [("8", (p_tag, c_word, c_tag))]
            keys += [("8", (p_tag, c_word, c_tag, direction, distance))]
            return keys

    def feature_9(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        direction, distance = self.get_direction_and_distance(p, c)
        p_word = self.hm_data[sen_idx]['words'][p]
        c_word = self.hm_data[sen_idx]['words'][c]
        c_tag = self.hm_data[sen_idx]['tags'][c]
        keys = [("9", (p_word, c_word, c_tag))]
        keys += [("9", (p_word, c_word, c_tag, direction, distance))]
        return keys

    def feature_10(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("10", (self.hm_data[sen_idx]['words'][p], self.hm_data[sen_idx]['tags'][p],
                            self.hm_data[sen_idx]['tags'][c]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            p_word = self.hm_data[sen_idx]['words'][p]
            p_tag = self.hm_data[sen_idx]['tags'][p]
            c_tag = self.hm_data[sen_idx]['tags'][c]
            keys = [("10", (p_word, p_tag, c_tag))]
            keys += [("10", (p_word, p_tag, c_tag, direction, distance))]
            return keys

    def feature_11(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        direction, distance = self.get_direction_and_distance(p, c)
        p_word = self.hm_data[sen_idx]['words'][p]
        p_tag = self.hm_data[sen_idx]['tags'][p]
        c_word = self.hm_data[sen_idx]['words'][c]
        keys = [("11", (p_word, p_tag, c_word))]
        keys += [("11", (p_word, p_tag, c_word, direction, distance))]
        return keys

    def feature_12(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        direction, distance = self.get_direction_and_distance(p, c)
        p_word = self.hm_data[sen_idx]['words'][p]
        c_word = self.hm_data[sen_idx]['words'][c]
        keys = [("12", (p_word, c_word))]
        keys += [("12", (p_word, c_word, direction, distance))]
        return keys

    def feature_13(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        if self.model == Consts.BASIC_MODEL:
            return [("13", (self.hm_data[sen_idx]['tags'][p], self.hm_data[sen_idx]['tags'][c]))]
        else:
            direction, distance = self.get_direction_and_distance(p, c)
            keys = [("13", (self.hm_data[sen_idx]['tags'][p], self.hm_data[sen_idx]['tags'][c]))]
            keys += [("13", (self.hm_data[sen_idx]['tags'][p], self.hm_data[sen_idx]['tags'][c], direction, distance))]
            return keys

    def feature_tags_between(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        min_word_idx = min(p, c)
        max_word_idx = max(p, c)
        keys = []
        direction, distance = self.get_direction_and_distance(p, c)
        p_tag = self.hm_data[sen_idx]['tags'][p]
        c_tag = self.hm_data[sen_idx]['tags'][c]
        for word_idx in range(min_word_idx + 1, max_word_idx):
            keys += [("tags_between", (p_tag, self.hm_data[sen_idx]['tags'][word_idx], c_tag))]
            keys += [("tags_between", (p_tag, self.hm_data[sen_idx]['tags'][word_idx], c_tag, direction, distance))]
        return keys

    def feature_contextual_tags(self, sen_idx: int, hm: tuple):
        p = hm[0]
        c = hm[1]
        keys = []
        direction, distance = self.get_direction_and_distance(p, c)
        p_tag = self.hm_data[sen_idx]['tags'][p]
        c_tag = self.hm_data[sen_idx]['tags'][c]
        for p_shift, c_shift in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
            if 0 <= p + p_shift < len(self.hm_data[sen_idx]['tags']) and \
               0 <= c + c_shift < len(self.hm_data[sen_idx]['tags']):
                p_shift_tag = self.hm_data[sen_idx]['tags'][p + p_shift]
                c_shift_tag = self.hm_data[sen_idx]['tags'][c + c_shift]
                keys += [("contextual_tags", (p_tag, p_shift_tag, c_tag, c_shift_tag))]
                keys += [("contextual_tags", (p_tag, p_shift_tag, c_tag, c_shift_tag, direction, distance))]

                keys += [("contextual_tags", (p_tag, p_shift_tag, c_tag, direction, distance))] + \
                        [("contextual_tags", (p_tag, c_tag, c_shift_tag, direction, distance))]
                keys += [("contextual_tags", (p_tag, p_shift_tag, c_tag))] + \
                        [("contextual_tags", (p_tag, c_tag, c_shift_tag))]
        return keys

    def get_features_idx_per_h_m(self, sen_idx, hm):
        features_idx = []
        for feature in self.used_features:
            keys_per_feature = self.features_funcs[feature](sen_idx, hm)
            for keys in keys_per_feature:
                if self.feature_vector.get(keys):
                    features_idx.append(self.feature_vector[keys][0])
        return features_idx

    # Saves for each sentence and (h, m) a list of the features its light
    def _calculate_features_for_all_sen_hm(self):
        Consts.print_info("_calculate_features_for_all_sen_hm", "Preprocessing")
        Consts.TIME = 1
        t1 = time()
        self.sentence_hm = {}

        for sen_idx, sentence in enumerate(self.hm_data):
            for i in range(len(sentence['words'])-1):
                for j in range(1, len(sentence['words'])-1):
                    if i == j:
                        continue
                    self.sentence_hm[(sen_idx, (i, j))] = self.get_features_idx_per_h_m(sen_idx, (i, j))

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

    def count_features_types(self):
        for feature_type in self.used_features:
            count_feature = sum([1 if feature_num == feature_type else 0 for feature_num, _ in self.feature_vector])
            Consts.print_info("feature_" + feature_type, "Has " + str(count_feature) + " features")
