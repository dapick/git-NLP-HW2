class Consts:

    TABLE_IDX = {'WORD_IDX': 0, 'WORD': 1, 'POS': 3, 'HEAD_IDX': 6, 'UNDERSCORE': [2, 4, 5, 7, 8, 9]}

    TRAIN = "Train"
    LABEL = "Label"

    PATH_TO_TRAINING = "HW2-files/train.labeled"

    @staticmethod
    def print_info(function_name: str, message: str):
        print("-I-(" + function_name + "): " + message)

