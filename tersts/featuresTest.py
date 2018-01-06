import unittest

from features import Features
from consts import Consts


class FeaturesTestCase(unittest.TestCase):
    def test_count_features_types(self):
        feature = Features(Consts.TRAIN, Consts.BASIC_MODEL, Consts.FEATURE_LIST_BASIC, Consts.PATH_TO_TRAINING_FROM_TEST)
        feature.print_features_to_file()


if __name__ == '__main__':
    unittest.main()
