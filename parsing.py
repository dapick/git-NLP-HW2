from consts import Consts


class Parsing:
    def __init__(self):
        self.column = Consts.TABLE_IDX
        self.labeled_table_data = []
        self.unlabeled_table = []

    # Creates list of sentences. each sentence holds dictionary:
    # labeled_table_data[idx] = { 'words' => [ , , ... ],
    #                     'tags' => [ , , ... ],
    #                     'heads'  => [ , , ... ],
    #                     'edges' => set((h0,m0), (h1,m1), ....) }
    def parse_labeled_file_to_list_of_dict(self, file_full_name: str):
        with open(file_full_name, 'r') as f:
            lines = f.readlines()

        lines_in_table = [line.split() for line in lines]
        sentence_idx = -1
        for line in lines_in_table:
            if not line:
                # Adds 'end_sentence' to mark the ending of the sentence
                if self.labeled_table_data[sentence_idx]['words'][-1] != 'end_sentence':
                    self.labeled_table_data[sentence_idx]['words'].append('end_sentence')
                    self.labeled_table_data[sentence_idx]['tags'].append('STOP')
                    self.labeled_table_data[sentence_idx]['heads'].append(len(self.labeled_table_data[sentence_idx]['heads']))
                continue
            if line[self.column['WORD_IDX']] == "1":
                sentence_idx += 1
                self.labeled_table_data.append({})
                self.labeled_table_data[sentence_idx]['words'] = ["root"]
                self.labeled_table_data[sentence_idx]['tags'] = ["*"]
                self.labeled_table_data[sentence_idx]['heads'] = [0]
                self.labeled_table_data[sentence_idx]['edges'] = set()

            self.labeled_table_data[sentence_idx]['words'].append(line[self.column['WORD']])
            self.labeled_table_data[sentence_idx]['tags'].append(line[self.column['POS']])
            self.labeled_table_data[sentence_idx]['heads'].append(int(line[self.column['HEAD_IDX']]))
            h = int(line[self.column['HEAD_IDX']])
            m = int(line[self.column['WORD_IDX']])
            self.labeled_table_data[sentence_idx]['edges'] |= {(h, m)}

        return self.labeled_table_data

    def parse_list_of_dict_to_labeled_file(self, file_full_name: str, words_tags: list, heads: list):
        with open(file_full_name, 'w+') as f:
            for sen_idx, sentence in enumerate(words_tags):
                for idx in range(1, len(sentence['words'])-1):
                    line = str(idx) + "\t" + sentence['words'][idx] + "\t_\t" + sentence['tags'][idx] + \
                           "\t_\t_\t" + str(heads[sen_idx][idx]) + "\t_\t_\t_\n"
                    f.write(line)
                f.write("\n")

    @staticmethod
    def parse_labeled_file_to_unlabeled_file(file_full_name: str):
        tmp_parse = Parsing()
        tmp_parse.parse_labeled_file_to_list_of_dict(file_full_name)
        file_unlabeled = file_full_name.replace('.labeled', '.unlabeled')
        with open(file_unlabeled, 'w+') as f:
            for sentence in tmp_parse.labeled_table_data:
                for idx in range(1, len(sentence['words'])-1):
                    line = str(idx) + "\t" + sentence['words'][idx] + "\t_\t" + sentence['tags'][idx] +\
                           "\t_\t_\t_\t_\t_\t_\n"
                    f.write(line)
                f.write("\n")

    # Creates list of sentences. each sentence holds a dictionary:
    # unlabeled_table[idx] = { 'words'  => [ , , ... ],
    #                          'tags'   => [ , , ... ] }
    def parse_unlabeled_file_to_list_of_dict(self, file_full_name: str):
        with open(file_full_name, 'r') as f:
            lines = f.readlines()

        lines_in_table = [line.split() for line in lines]
        sentence_idx = -1
        for line in lines_in_table:
            if not line:
                # Adds 'end_sentence' to mark the ending of the sentence
                if self.unlabeled_table[sentence_idx]['words'][-1] != 'end_sentence':
                    self.unlabeled_table[sentence_idx]['words'].append('end_sentence')
                    self.unlabeled_table[sentence_idx]['tags'].append('STOP')
                continue
            if line[self.column['WORD_IDX']] == "1":
                sentence_idx += 1
                self.unlabeled_table.append({})
                self.unlabeled_table[sentence_idx]['words'] = ["root"]
                self.unlabeled_table[sentence_idx]['tags'] = ["*"]

            self.unlabeled_table[sentence_idx]['words'].append(line[self.column['WORD']])
            self.unlabeled_table[sentence_idx]['tags'].append(line[self.column['POS']])

        return self.unlabeled_table
