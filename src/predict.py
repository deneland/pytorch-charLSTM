import sys
import random
from model import *
from data import *
import torch.nn as nn


def get_input_tensor(line, character_lookup):
    indexes = torch.LongTensor([character_lookup[c] for c in line])
    tensor = nn.functional.one_hot(indexes, len(list(character_lookup.keys())))
    tensor = torch.unsqueeze(tensor, 0)
    return tensor


def sample(net, primer_sequence, character_lookup, batch_size, max_len=100):
    with torch.no_grad():

        hidden = net.init_state(batch_size)
        hidden = tuple([h.to(net.device) for h in hidden])

        for prime_char in primer_sequence:
            output = get_input_tensor(prime_char, character_lookup)[0].to(net.device)
            output, hidden = net(output, hidden)
        
        result = primer_sequence

        while True:
            output, hidden = net(output, hidden)

            weights = torch.exp(output.squeeze())
            next_char = random.choices(list(character_lookup.keys()), weights=weights)[0]
            if len(result) > max_len:
                return result

            result += next_char
            output = get_input_tensor(result[-1], character_lookup)[0].to(net.device)


if __name__ == "__main__":
    import json
    import yaml

    with open(sys.argv[2], "r") as f:
        vocabulary = json.load(f)

    model_params = yaml.safe_load(open("params.yaml"))["model"]

    characters = list(vocabulary.keys())
    n_characters = len(characters)

    net = LanguageModel(n_characters, torch.device("cpu"), model_params)

    net.load_state_dict(torch.load(sys.argv[1]))
    net.eval()

    if len(sys.argv) > 3:
        first_char = sys.argv[3]
    else:
        first_char = random.choice(characters)

    generated = sample(net, first_char, vocabulary, 1, max_len=500)
    print(generated)

