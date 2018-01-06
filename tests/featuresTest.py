import unittest

from features import Features
from consts import Consts


class FeaturesTestCase(unittest.TestCase):
    def test_count_features_types(self):
        feature = Features(Consts.TRAIN, Consts.BASIC_MODEL, Consts.FEATURE_LIST_BASIC, Consts.PATH_TO_TRAINING_FROM_TEST)
        feature.print_features_to_file()

    def test_match_features_to_hm(self):
        feature = Features(Consts.LABEL, Consts.BASIC_MODEL, Consts.FEATURE_LIST_BASIC,
                           Consts.PATH_TO_TRAINING_FROM_TEST)
        print(feature.get_features_idx_per_h_m(0, (2, 7)))


if __name__ == '__main__':
    unittest.main()
