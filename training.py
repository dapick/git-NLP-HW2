from consts import Consts
from features import Features


class Training:

    def __init__(self, model: str, used_features: list, file_full_name: str):
        self.feature = Features(Consts.TRAIN, model, used_features, file_full_name)