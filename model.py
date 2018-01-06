from consts import Consts
from training import Training
from features import Features

import pickle
import abc


class Model(metaclass=abc.ABCMeta):
    feature = None
    w_parameter = None

    def __init__(self, method: str, N: int, file_full_name: str=None):
        if method == Consts.TRAIN:
            self._training(N, file_full_name)
        elif method == Consts.LABEL:
            self._set_internal_values(N, file_full_name)

    @abc.abstractmethod
    def _training(self, N: int, file_full_name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_internal_values(self, N: int, file_full_name: str):
        raise NotImplementedError


class BasicModel(Model):
    def __init__(self, method: str, N: int, file_full_name: str=Consts.PATH_TO_TRAINING):
        super().__init__(method, N, file_full_name)

    def _training(self, N: int, file_full_name: str):
        self.w_parameter = Training(Consts.BASIC_MODEL, N, file_full_name, Consts.FEATURE_LIST_BASIC.w_parameter)

    def _set_internal_values(self, N: int, file_full_name: str):
        self.feature = Features(Consts.LABEL, Consts.BASIC_MODEL, N, file_full_name)
