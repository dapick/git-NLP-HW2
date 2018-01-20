import unittest

from inference import Inference
from shutil import copy
from consts import Consts


class MyTestCase(unittest.TestCase):
    # TODO: remove other prints before submission and extract from tests and changed the path accordingly
    def test_label_comp_by_basic(self):
        N = 50
        inference = Inference(Consts.PATH_TO_COMPETITION_FROM_TEST, Consts.BASIC_MODEL, N)
        print("Finished labeling comp file by 'basic'")
        copy(inference.labeled_file_name, "../data_from_training/" + Consts.BASIC_MODEL + "/" + str(N))

    def test_label_comp_by_advanced(self):
        N = 50
        inference = Inference(Consts.PATH_TO_COMPETITION_FROM_TEST, Consts.ADVANCED_MODEL, N)
        print("Finished labeling comp file by 'basic'")
        copy(inference.labeled_file_name, "../data_from_training/" + Consts.ADVANCED_MODEL + "/" + str(N))


if __name__ == '__main__':
    unittest.main()
