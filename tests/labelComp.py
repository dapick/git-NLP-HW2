import unittest

from inference import Inference
from consts import Consts


class MyTestCase(unittest.TestCase):
    # TODO: remove other prints before submission and extract from tests and changed the path accordingly
    def test_label_comp_by_basic(self):
        N = 50
        Inference("../HW2-files/comp.unlabeled", Consts.BASIC_MODEL, N)

    def test_label_comp_by_advanced(self):
        N = 50
        Inference("../HW2-files/comp.unlabeled", Consts.ADVANCED_MODEL, N)


if __name__ == '__main__':
    unittest.main()
