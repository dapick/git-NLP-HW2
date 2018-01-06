import unittest
from parsing import Parsing


class MyTestCase(unittest.TestCase):
    def test_something(self):
        p = Parsing()
        p.parse_labeled_file_to_list_of_dict("HW2-files/test.labeled")
        # p.parse_list_of_dict_to_labeled_file("HW2-files/test.labeled.test")
        Parsing.parse_test_file_to_unlabeled_file("HW2-files/test.labeled")


if __name__ == '__main__':
    unittest.main()
