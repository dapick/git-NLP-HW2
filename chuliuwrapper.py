class ChuLiuWrapper:
    def __init__(self, sentences):

        self.sentences = sentences
        self.sentences_klicks = self.init_klicks()

    def init_klicks(self):
        list_of_klicks = []
        for idx, sentence in enumerate(self.sentences):
            list_of_klicks.append({})
            list_of_klicks[idx][0] = list(range(1, len(sentence['words'])))
            for i in range(1, len(sentence['words'])):
                list_of_klicks[idx][i] = list((range(1, i))) + list(range(i + 1, len(sentence['words'])))

        return list_of_klicks

    def print_klick_to_file(self):
        with open('../data_from_training/kilck', 'w+') as f:
            for idx, sentence in enumerate(self.sentences_klicks):
                f.write(str(idx) + " =>\n")
                for key, value in sentence.items():
                    f.write("\t" + str(key) + " => " + str(value) + "\n")
