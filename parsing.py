from consts import Consts


class Parsing:
    def __init__(self):
        self.column = Consts.TABLE_IDX
        self.table_data = []

    # creates list of sentences. each sentence holds dictionary:
    # table_data[idx] = { 'words' => [ , , ... ],
    #                     'tags' => [ , , ... ],
    #                     'heads'  => [ , , ... ],
    #                     'edges' => set((h0,m0), (h1,m1), ....) }
    def parse_file_to_list_of_dict(self, file_full_name: str):
        with open(file_full_name, 'r') as f:
            lines = f.readlines()

        lines_in_table = [line.split() for line in lines]
        sentence_idx = -1
        for line in lines_in_table:
            if not line:
                continue
            if line[self.column['WORD_IDX']] == "1":
                sentence_idx += 1
                self.table_data.append({})
                self.table_data[sentence_idx]['words'] = ["root"]
                self.table_data[sentence_idx]['tags'] = ["*"]
                self.table_data[sentence_idx]['heads'] = ["*"]
                self.table_data[sentence_idx]['edges'] = set()

            self.table_data[sentence_idx]['words'].append(line[self.column['WORD']])
            self.table_data[sentence_idx]['tags'].append(line[self.column['POS']])
            self.table_data[sentence_idx]['heads'].append(int(line[self.column['HEAD_IDX']]))
            h = int(line[self.column['HEAD_IDX']])
            m = int(line[self.column['WORD_IDX']])
            self.table_data[sentence_idx]['edges'] |= {(h, m)}

    def parse_list_of_dict_to_file(self, file_full_name: str):
        with open(file_full_name, 'w+') as f:
            for sentence in self.table_data:
                for idx in range(1, len(sentence['words'])):
                    line = str(idx) + "\t" + sentence['words'][idx] + "\t_\t" + sentence['tags'][idx] +\
                           "\t_\t_\t" + str(sentence['heads'][idx]) + "\t_\t_\t_\n"
                    f.write(line)
                f.write("\n")
