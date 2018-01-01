from parsing import Parsing
from consts import Consts


class Features:

    def __init__(self, method: str, model: str, used_features: list=None, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(model, used_features, file_full_name)

        elif method == Consts.LABEL:
            self.list_labeled_data = Parsing().parse_unlabeled_file_to_list_of_dict(file_full_name)

    def _training(self, model: str, used_features: list, file_full_name: str=Consts.PATH_TO_TRAINING):
        self.features_funcs = {"1": self.feature_1, "2": self.feature_2, "3": self.feature_3, "4": self.feature_4,
                               "5": self.feature_5, "6": self.feature_6, "7": self.feature_7, "8": self.feature_8,
                               "9": self.feature_9, "10": self.feature_10, "11": self.feature_11, "12": self.feature_12,
                               "13": self.feature_13}

        self.used_features = used_features
        self.idx = 0
        self.feature_vector = {}
        self.features_occurrences = []
        self.list_labeled_data = Parsing().parse_labeled_file_to_list_of_dict(file_full_name)


    # Gives an index for each feature and count how many time it was used
    def feature_structure(self, keys: tuple):
        if keys not in self.feature_vector:
            self.feature_vector[keys] = [self.idx, 1]
            self.features_occurrences.append(1)
            self.idx += 1
        else:
            self.feature_vector[keys][1] += 1
            self.features_occurrences[self.feature_vector[keys][0]] += 1

    def feature_1(self):
        Consts.print_info("feature_1", "Building")
        for sentence in self.list_labeled_data:
            for idx, word in enumerate(sentence['words']):
                p = sentence['heads'][idx]
                p_word = sentence['words'][p]
                p_pos = sentence['tags'][p]
                self.feature_structure(("1", (p_word, p_pos)))

    def feature_2(self):
        Consts.print_info("feature_2", "Building")
        for sentence in self.list_labeled_data:
            for idx, word in enumerate(sentence['words']):
                p = sentence['heads'][idx]
                p_word = sentence['words'][p]
                self.feature_structure(("2", p_word))

    def feature_3(self):
        Consts.print_info("feature_3", "Building")
        for sentence in self.list_labeled_data:
            for idx, word in enumerate(sentence['words']):
                p = sentence['heads'][idx]
                p_pos = sentence['tags'][p]
                self.feature_structure(("3", p_pos))

    def feature_4(self):
        Consts.print_info("feature_4", "Building")
        for sentence in self.list_labeled_data:
            for h_m in sentence['edges']:
     


