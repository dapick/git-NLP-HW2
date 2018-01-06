from chuliuwrapper import ChuLiuWrapper
from parsing import Parsing
from consts import Consts
from model import BasicModel
from chu_liu import Digraph

from multiprocessing.pool import Pool
from time import time
import numpy as np
from functools import partial


class Inference(object):
    def __init__(self, file_full_name: str, model: str):
        self.file_full_name = file_full_name
        self.labeled_file_name = self._get_labeled_file_name()
        self.model_type = model
        self.successors_per_sentence = ChuLiuWrapper(self.model.feature.hm_data).sentences_klicks

        if model == Consts.BASIC_MODEL:
            self.model = BasicModel(Consts.LABEL, file_full_name)
        # elif model == Consts.ADVANCED_MODEL:
        #     self.model = AdvancedModel(Consts.TAG)

    def _get_labeled_file_name(self):
        return self.file_full_name.replace('unlabeled', 'labeled')

    def get_score(self, sen_idx: int, h: int, m: int):
        return np.sum(self.model.w_parameter[self.model.feature.sentence_hm[(sen_idx, (h, m))]])

    def _label_sentence(self, sen_idx):
        y_max = Digraph(self.successors_per_sentence[sen_idx], partial(self.get_score, sen_idx)).mst().successors
        heads = np.zeros(len(self.model.feature.hm_data[sen_idx]['words']))
        for head in y_max:
            for modifier in y_max[head]:
                heads[modifier] = head
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
