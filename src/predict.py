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


def sample(net, init_char, character_lookup, batch_size):
    with torch.no_grad():
        result = init_char

        input = get_input_tensor(result[-1], character_lookup)[0].to(net.device)

        hidden = net.init_state(batch_size)
        hidden = tuple([h.to(net.device) for h in hidden])

        while True:
            output, hidden = net(input, hidden)
            next_char = decode_tensor(output[0], list(character_lookup.keys()), topk=1)
            if next_char == "EOL" or len(result) > 100:
                return result

            result += next_char
            input = get_input_tensor(result[-1], character_lookup)[0].to(net.device)


if __name__ == "__main__":
    import json
    import yaml

    with open(sys.argv[2], "r") as f:
        vocabulary = json.load(f)

    model_params = yaml.safe_load(open("params.yaml"))["model"]

    characters = list(vocabulary.keys())
    n_characters = len(characters)

    net = LanguageModel(n_characters, model_params, torch.device("cpu"))

    net.load_state_dict(torch.load(sys.argv[1]))
    net.eval()

    if len(sys.argv) > 3:
        first_char = sys.argv[3]
    else:
        first_char = random.choice(characters)

    generated = sample(net, first_char, vocabulary, 1)
    print(generated)
