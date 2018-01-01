import unittest
from parsing import Parsing


class MyTestCase(unittest.TestCase):
    def test_something(self):
        p = Parsing()
        p.parse_file_to_list_of_dict("HW2-files/test.labeled")
        p.parse_list_of_dict_to_file("HW2-files/test.labeled.test")


if __name__ == '__main__':
    unittest.main()
