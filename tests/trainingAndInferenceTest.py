import unittest

from training import Training
from inference import Inference
from consts import Consts


class MyTestCase(unittest.TestCase):
    def test_5_sentences_for_w(self):
        train = Training(Consts.BASIC_MODEL, Consts.PATH_TO_5_SENTENCES, Consts.FEATURE_LIST_BASIC)
        train.feature.print_features_to_file('smaller_sentences/')
        print(train.w_parameter)

    def test_all_sentences_for_w(self):
        train = Training(Consts.BASIC_MODEL, Consts.PATH_TO_TRAINING_FROM_TEST, Consts.FEATURE_LIST_BASIC)

    def test_inference_for_5_sentences(self):
        inference = Inference('smaller_sentences/5_sentences.unlabeled', Consts.BASIC_MODEL)
        inference.label()


if __name__ == '__main__':
    unittest.main()
