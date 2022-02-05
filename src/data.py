import torch
import unicodedata
import string
import random

# characters = string.printable
# characters = 'abcdefghijklmnopqrstuvwxyz'
characters = string.ascii_letters + " .,;'-"
character_lookup = {character: i for i, character in enumerate(characters)}
character_lookup["EOL"] = len(character_lookup.keys())
n_characters = len(character_lookup.keys())


def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in characters
    )


def character_onehot(character):
    onehot = [0 for _ in range(len(character_lookup.keys()))]
    onehot[character_lookup[character]] = 1

    return onehot

def clean_line(line):
    line = line.strip()
    # line = line.lower()
    line = unicodeToAscii(line)

    return line

def get_input_tensor(line):
    onehot_matrix = [character_onehot(c) for c in line]
    # onehot_matrix.append(character_onehot("EOL"))
    tensor = torch.tensor(onehot_matrix)
    tensor = torch.unsqueeze(tensor, 1)

    return tensor

def get_target_tensor(line):
    letter_indexes = [character_lookup[char] for i, char in enumerate(line) if i != 0]
    letter_indexes.append(character_lookup['EOL'])
    return torch.LongTensor(letter_indexes)

def decode_tensor(tensor, topk=1):
    try:
        _, topi = tensor.topk(topk)
        topi = random.choice(topi)
        next_char = characters[topi]
    except IndexError:
        next_char = "EOL"

    return next_char