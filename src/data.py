from calendar import c
import torch
import unicodedata
import string
import random
from torch.utils.data import Dataset

def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in characters
    )


def character_onehot(character, character_lookup):
    onehot = [0 for _ in range(len(character_lookup.keys()))]
    onehot[character_lookup[character]] = 1

    return onehot


def clean_line(line):
    line = line.strip()
    # line = line.lower()
    line = unicodeToAscii(line)

    return line


def decode_tensor(tensor, characters, topk=1):
    try:
        _, topi = tensor.topk(topk)
        topi = random.choice(topi)
        next_char = characters[topi]
    except IndexError:
        next_char = "EOL"

    return next_char


def get_entry(text, text_length, segment_length):
    start = random.randint(0, text_length - segment_length)
    batch = text[start : start + segment_length]

    return batch


def get_batch(batch_size, text, text_length, segment_length):
    return [get_entry(text, text_length, segment_length) for _ in range(batch_size)]



class TextDataset(Dataset):
    def __init__(self, text_file, segment_length, character_lookup=None):
        with open(text_file, "r") as f:
            self.text = "".join(f.readlines())

        if character_lookup is None:
            characters = list(set(self.text))
            self.character_lookup = {character: i for i, character in enumerate(characters)}
        else:
            character_lookup = character_lookup

        self.segment_length = segment_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        start = random.randint(0, len(self) - self.segment_length)
        line = self.text[start : start + self.segment_length]

        input = self.get_input_tensor(line)
        output = self.get_input_tensor(line)

        return input, output

    def get_input_tensor(self, line):
        onehot_matrix = [character_onehot(c, self.character_lookup) for c in line]
        # onehot_matrix.append(character_onehot("EOL"))
        tensor = torch.tensor(onehot_matrix)
        # tensor = torch.unsqueeze(tensor, 0)

        return tensor


    def get_target_tensor(self, line):
        letter_indexes = [self.character_lookup[char] for i, char in enumerate(line) if i != 0]
        letter_indexes.append(self.character_lookup["EOL"])
        tensor = torch.LongTensor(letter_indexes)
        # tensor = torch.unsqueeze(tensor, 0)

        return tensor


if __name__ == '__main__':
    ds = TextDataset('data/texts/shakespeare.txt', 300)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(ds, batch_size=4, shuffle=False)

    for i in range(1):
        batch = next(iter(dataloader))

        print(batch[0].shape)
        print(batch[1].shape)



