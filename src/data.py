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
    def __init__(self, segments, vocabulary):

        self.n_char = len(list(vocabulary.keys()))
        self.segment_length = len(segments[0])

        self.indexes = torch.LongTensor([[vocabulary[c] for c in segment] for segment in segments])
        self.onehot = nn.functional.one_hot(self.indexes, self.n_char)
        self.indexes = self.indexes[:, 1:]
        self.onehot = self.onehot[:, :-1, :]

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, idx):
        input = self.onehot[idx]
        output = self.indexes[idx]

        return input, output

def get_datasets(text_file, vocabulary, params):
    with open(text_file, "r") as f:
        text = "".join(f.readlines())

    segments = []
    start_idx = 0
    end_idx = params['segment_length']
    while end_idx <= len(text):
        segments.append(text[start_idx:end_idx])
        start_idx += params['segment_length']
        end_idx += params['segment_length']
    
    random.shuffle(segments)
    n_train = int(len(segments)*params['train_split'])
    train_segments = segments[:n_train]
    val_segments = segments[n_train:]

    ds_train = TextDataset(train_segments, vocabulary)
    ds_val = TextDataset(val_segments, vocabulary)

    return ds_train, ds_val


if __name__ == "__main__":
    import sys
    import json
    import yaml

    with open('vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    params = yaml.safe_load(open("params.yaml"))["train"]
    ds_train, ds_val = get_datasets(sys.argv[1], vocabulary, params)

    input, target = ds_train[0]

    for ch_idx in range(target.shape[0]):
        print(decode_tensor(input[ch_idx, :], list(vocabulary.keys())), list(vocabulary.keys())[target[ch_idx]])