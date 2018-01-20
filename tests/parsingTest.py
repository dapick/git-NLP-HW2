import unittest
from parsing import Parsing


class MyTestCase(unittest.TestCase):
    def test_parsing(self):
        p = Parsing()
        # p.parse_labeled_file_to_list_of_dict("HW2-files/test.labeled")
        # p.parse_list_of_dict_to_labeled_file("HW2-files/test.labeled.test")
        # Parsing.parse_labeled_file_to_unlabeled_file("smaller_sentences/5_sentences.labeled")
        Parsing.parse_labeled_file_to_unlabeled_file("../HW2-files/comp_m1_302988217.wtag")


if __name__ == '__main__':
    unittest.main()
