from consts import Consts
from training import Training
from features import Features

import pickle
import abc


class Model(metaclass=abc.ABCMeta):
    feature = None
    w_parameter = None

    def __init__(self, method: str, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(file_full_name)
        elif method == Consts.LABEL:
            self._set_internal_values(file_full_name)

    @abc.abstractmethod
    def _training(self, file_full_name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_internal_values(self, file_full_name: str):
        raise NotImplementedError


class BasicModel(Model):
    def __init__(self, method: str, file_full_name: str=Consts.PATH_TO_TRAINING):
        super().__init__(method, file_full_name)

    def _training(self, file_full_name: str):
        self.w_parameter = Training(Consts.BASIC_MODEL, file_full_name, Consts.FEATURE_LIST_BASIC).w_parameter
        with open("../data_from_training/" + Consts.BASIC_MODEL + "/w_parameter", 'wb') as f:
            pickle.dump(self.w_parameter, f, protocol=-1)

    def _set_internal_values(self, file_full_name: str):
        with open("../data_from_training/" + Consts.BASIC_MODEL + "/w_parameter", 'rb') as f:
            self.w_parameter = pickle.load(f)
        self.feature = Features(Consts.LABEL, Consts.BASIC_MODEL, file_full_name)
