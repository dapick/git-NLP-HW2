class Consts:

    TABLE_IDX = {'WORD_IDX': 0, 'WORD': 1, 'POS': 3, 'HEAD_IDX': 6, 'UNDERSCORE': [2, 4, 5, 7, 8, 9]}

    TRAIN = "Train"
    LABEL = "Label"

    PATH_TO_TRAINING = "HW2-files/train.labeled"

    PATH_TO_TRAINING_FROM_TEST = "../HW2-files/train.labeled"
    PATH_TO_TEST_UNLABELED_FROM_TEST = "../HW2-files/test.unlabeled"

    # Paths for testing
    PATH_TO_5_SENTENCES = "smaller_sentences/5_sentences.labeled"

    BASIC_MODEL = "basic"
    ADVANCED_MODEL = "advanced"

    FEATURE_LIST_BASIC = ["1", "2", "3", "4", "5", "6", "8", "10", "13"]

    TIME = 0

    @staticmethod
    def print_info(function_name: str, message: str):
        print("-I-(" + function_name + "): " + message)

    @staticmethod
    def print_time(function_name: str, time: float):
        if Consts.TIME == 1:
            print("-T-(" + function_name + "): took " + str(time) + " seconds")
