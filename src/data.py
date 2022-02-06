import torch
import unicodedata
import string
import random


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

def get_input_tensor(line, character_lookup):
    onehot_matrix = [character_onehot(c, character_lookup) for c in line]
    # onehot_matrix.append(character_onehot("EOL"))
    tensor = torch.tensor(onehot_matrix)
    tensor = torch.unsqueeze(tensor, 0)

    return tensor

def get_target_tensor(line, character_lookup):
    letter_indexes = [character_lookup[char] for i, char in enumerate(line) if i != 0]
    letter_indexes.append(character_lookup['EOL'])
    tensor =  torch.LongTensor(letter_indexes)
    tensor = torch.unsqueeze(tensor, 0)

    return tensor

def decode_tensor(tensor, characters, topk=1):
    try:
        _, topi = tensor.topk(topk)
        topi = random.choice(topi)
        next_char = characters[topi]
    except IndexError:
        next_char = "EOL"

    return next_char

def get_entry(text, text_length, segment_length):
    start = random.randint(0, text_length-segment_length)
    batch = text[start:start+segment_length]

    return batch

def get_batch(batch_size, text, text_length, segment_length):
    return [get_entry(text, text_length, segment_length) for _ in range(batch_size)]
