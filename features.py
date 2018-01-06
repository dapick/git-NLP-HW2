from parsing import Parsing
from consts import Consts


class Features:

    def __init__(self, method: str, model: str, used_features: list=None, file_full_name: str=None):
        if method == Consts.TRAIN:
            self.model = model
            self._training(used_features, file_full_name)

        elif method == Consts.LABEL:
            self.hm_data = Parsing().parse_unlabeled_file_to_list_of_dict(file_full_name)

    def _training(self, used_features: list, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.used_features = used_features
        self.idx = 0
        self.feature_vector = {}
        self.hm_match_feature = []
        self.features_occurrences = []

        self.hm_data = Parsing().parse_labeled_file_to_list_of_dict(file_full_name)

        self._light_features()


    def _light_features(self):
        [self.hm_match_feature.append({}) for i in range(0, len(self.hm_data))]

        for feature in self.used_features:
            Consts.print_info("_light_features", "Building feature " + feature)
            for sen_idx, sentence in enumerate(self.hm_data):
                for h_m in sentence['edges']:
                    tup = self._feature_hm_match_data(sen_idx, h_m, feature)
                    self.feature_structure((feature, tup))
                    if h_m not in self.hm_match_feature[sen_idx]:
                        self.hm_match_feature[sen_idx][h_m] = set()
                    self.hm_match_feature[sen_idx][h_m].add(tup)



    # Gives an index for each feature and count how many time it was used
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

    def print_features_to_file(self):
        with open('../data_from_training/' + self.model + '/feature_vector', 'w+') as f:
            for key, values in self.feature_vector.items():
                f.write(str(key) + " => " + str(values) + "\n")

        with open('../data_from_training/' + self.model + '/h_m_match_to_feature', 'w+') as f:
            for i, x in enumerate(self.hm_match_feature):
                f.write(str(i) + " =>\n")
                for key, val in x.items():
                    f.write("\t" + str(key) + " => " + str(val) + "\n")


    # def feature_1(self):
    #     Consts.print_info("feature_1", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             p_word = sentence['words'][p]
    #             p_pos = sentence['tags'][p]
    #             self.feature_structure(("1", (p_word, p_pos)))
    #
    #
    # def feature_2(self):
    #     Consts.print_info("feature_2", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             p_word = sentence['words'][p]
    #             self.feature_structure(("2", p_word))
    #
    # def feature_3(self):
    #     Consts.print_info("feature_3", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             p_pos = sentence['tags'][p]
    #             self.feature_structure(("3", p_pos))
    #
    # def feature_4(self):
    #     Consts.print_info("feature_4", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             c = h_m[1]
    #             c_word = sentence['words'][c]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("4", (c_word, c_pos)))
    #
    # def feature_5(self):
    #     Consts.print_info("feature_5", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             c = h_m[1]
    #             c_word = sentence['words'][c]
    #             self.feature_structure(("5", c_word))
    #
    # def feature_6(self):
    #     Consts.print_info("feature_6", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             c = h_m[1]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("6", c_pos))
    #
    # def feature_7(self):
    #     Consts.print_info("feature_7", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_word = sentence['words'][p]
    #             p_pos = sentence['tags'][p]
    #             c_word = sentence['words'][c]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("7", (p_word, p_pos, c_word, c_pos)))
    #
    # def feature_8(self):
    #     Consts.print_info("feature_8", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_pos = sentence['tags'][p]
    #             c_word = sentence['words'][c]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("8", (p_pos, c_word, c_pos)))
    #
    # def feature_9(self):
    #     Consts.print_info("feature_9", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_word = sentence['words'][p]
    #             c_word = sentence['words'][c]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("9", (p_word, c_word, c_pos)))
    #
    # def feature_10(self):
    #     Consts.print_info("feature_10", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_word = sentence['words'][p]
    #             p_pos = sentence['tags'][p]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("10", (p_word, p_pos, c_pos)))
    #
    # def feature_11(self):
    #     Consts.print_info("feature_11", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_word = sentence['words'][p]
    #             p_pos = sentence['tags'][p]
    #             c_word = sentence['words'][c]
    #             self.feature_structure(("11", (p_word, p_pos, c_word)))
    #
    # def feature_12(self):
    #     Consts.print_info("feature_12", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_word = sentence['words'][p]
    #             c_word = sentence['words'][c]
    #             self.feature_structure(("12", (p_word, c_word)))
    #
    # def feature_13(self):
    #     Consts.print_info("feature_13", "Building")
    #     for sentence in self.hm_data:
    #         for h_m in sentence['edges']:
    #             p = h_m[0]
    #             c = h_m[1]
    #             p_pos = sentence['tags'][p]
    #             c_pos = sentence['tags'][c]
    #             self.feature_structure(("13", (p_pos, c_pos)))