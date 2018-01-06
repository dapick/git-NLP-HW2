import unittest

from features import Features
from parsing import Parsing
from chuliuwrapper import ChuLiuWrapper
from consts import Consts


class FeaturesTestCase(unittest.TestCase):

    def test_klicks_init(self):
        data = Parsing()
        wrapper = ChuLiuWrapper(data.parse_unlabeled_file_to_list_of_dict(Consts.PATH_TO_TEST_UNLABELED_FROM_TEST))
        wrapper.print_klick_to_file()

if __name__ == '__main__':
    unittest.main()
