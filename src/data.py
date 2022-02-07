from calendar import c
import torch
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
            text = "".join(f.readlines())

        if character_lookup is None:
            characters = list(set(text))
            self.character_lookup = {
                character: i for i, character in enumerate(characters)
            }
        else:
            character_lookup = character_lookup

        self.n_char = len(list(self.character_lookup.keys()))
        self.segment_length = segment_length

        self.indexes = torch.LongTensor([self.character_lookup[c] for c in text])
        self.onehot = nn.functional.one_hot(self.indexes, self.n_char)

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, idx):
        start = random.randint(0, len(self) - self.segment_length)

        input = self.onehot[start : start + self.segment_length]
        output = self.indexes[start + 1 : start + self.segment_length]

        return input, output


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    ds = TextDataset(sys.argv[1], int(sys.argv[2]))
    print(ds.character_lookup)

    dataloader = DataLoader(ds, batch_size=4, shuffle=False)

    for i in range(1):
        batch = next(iter(dataloader))

        print(batch[0].shape)
        print(batch[1].shape)
