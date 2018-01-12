import unittest

from features import Features
from consts import Consts


class FeaturesTestCase(unittest.TestCase):
    def test_count_features_types_for_each_model(self):
        # TODO: before running put in comment unnecessary commands in feature that just takes time
        feature = Features(Consts.TRAIN, Consts.BASIC_MODEL, 20, Consts.PATH_TO_TRAINING_FROM_TEST,
                           Consts.FEATURE_LIST_BASIC)
        print("BASIC:")
        feature.count_features_types()

        print()

        feature = Features(Consts.TRAIN, Consts.ADVANCED_MODEL, 20, Consts.PATH_TO_TRAINING_FROM_TEST,
                           Consts.FEATURE_LIST_ADVANCED)
        print("ADVANCED:")
        feature.count_features_types()

    def test_match_features_to_hm(self):
        feature = Features(Consts.LABEL, Consts.BASIC_MODEL, 20, Consts.FEATURE_LIST_BASIC,
                           Consts.PATH_TO_TRAINING_FROM_TEST)
        print(feature.get_features_idx_per_h_m(0, (2, 7)))


if __name__ == '__main__':
    unittest.main()
