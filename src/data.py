from calendar import c
import torch
import unicodedata
import string
import random
from torch.utils.data import Dataset
import torch.nn as nn


def decode_tensor(tensor, characters, topk=1):
    try:
        _, topi = tensor.topk(topk)
        topi = random.choice(topi)
        next_char = characters[topi]
    except IndexError:
        next_char = "EOL"

    return next_char


class TextDataset(Dataset):
    def __init__(self, text_file, segment_length, character_lookup=None):
        with open(text_file, "r") as f:
            self.text = "".join(f.readlines())

        if character_lookup is None:
            characters = list(set(self.text))
            self.character_lookup = {character: i for i, character in enumerate(characters)}
        else:
            character_lookup = character_lookup

        self.n_char = len(list(self.character_lookup.keys()))
        self.segment_length = segment_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        start = random.randint(0, len(self) - self.segment_length)
        line = self.text[start : start + self.segment_length]

        input = self.get_input_tensor(line)
        output = self.get_target_tensor(line)

        return input, output

    def get_input_tensor(self, line):
        indexes = torch.LongTensor([self.character_lookup[c] for c in line])
        tensor = nn.functional.one_hot(indexes, self.n_char)

        return tensor

    def get_target_tensor(self, line):
        indexes = [self.character_lookup[c] for c in line[1:]]
        tensor = torch.LongTensor(indexes)

        return tensor


if __name__ == '__main__':
    ds = TextDataset('data/texts/shakespeare.txt', 300)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(ds, batch_size=4, shuffle=False)

    for i in range(1):
        batch = next(iter(dataloader))

        print(batch[0].shape)
        print(batch[1].shape)



