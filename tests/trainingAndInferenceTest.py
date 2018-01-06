import unittest

from training import Training
from inference import Inference
from consts import Consts
from time import time


class MyTestCase(unittest.TestCase):
    # TODO: before running these tests, make sure to back the original files because it will override them
    # def test_5_sentences_for_w(self):
    #     t1 = time()
    #     train = Training(Consts.BASIC_MODEL, 50, Consts.PATH_TO_5_SENTENCES, Consts.FEATURE_LIST_BASIC)
    #     with open('runningTimes', 'a') as f:
    #         print("Took " + str(time() - t1) + " seconds to train with 20 iterations", file=f)
    #     train.feature.print_features_to_file('smaller_sentences/')
    #     print(train.w_parameter)
    #
    # def test_inference_for_5_sentences(self):
    #     inference = Inference('smaller_sentences/5_sentences.unlabeled', Consts.BASIC_MODEL, 20)
    #     inference.label()
    #     inference.calculate_accuracy(inference.labeled_file_name, 'smaller_sentences/5_sentences.labeled')

    def test_all_sentences_for_w(self):
        train = Training(Consts.BASIC_MODEL, 20, Consts.PATH_TO_TRAINING_FROM_TEST, Consts.FEATURE_LIST_BASIC)

    def test_inference_for_1000_sentences(self):
        inference = Inference(Consts.PATH_TO_TEST_UNLABELED_FROM_TEST, Consts.BASIC_MODEL, 20)
        inference.label()
        inference.calculate_accuracy(inference.labeled_file_name, Consts.PATH_TO_TEST_LABELED_FROM_TEST)

    def test_20_iterations(self):
        t1 = time()
        Training(Consts.BASIC_MODEL, 20, Consts.PATH_TO_TRAINING_FROM_TEST, Consts.FEATURE_LIST_BASIC)
        with open('runningTimes', 'a') as f:
            print("Took " + str(time() - t1) + " seconds to train with 20 iterations", file=f)
        t1 = time()
        inference = Inference(Consts.PATH_TO_TEST_UNLABELED_FROM_TEST, Consts.BASIC_MODEL, 20)
        inference.label()
        with open('runningTimes', 'a') as f:
            print("Took " + str(time() - t1) + " seconds to label with 20 iterations", file=f)
        inference.calculate_accuracy(inference.labeled_file_name, Consts.PATH_TO_TEST_LABELED_FROM_TEST)


if __name__ == '__main__':
    unittest.main()
