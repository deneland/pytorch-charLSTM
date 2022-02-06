import re
import sys
from dataclasses import dataclass
import torch

import random
import os

from model import *
from data import *
from predict import *

import time
import math
import yaml
import json

def train(net, batch, device, params):
    criterion = torch.nn.NLLLoss()

    hidden = net.init_state(params['batch_size'])#.to(device)
    net.zero_grad()

    loss = 0
    input_tensor = [get_input_tensor(line, character_lookup) for line in batch]
    input_tensor = torch.cat(tuple(input_tensor), 0)

    target_tensor = [get_target_tensor(line, character_lookup) for line in batch]
    target_tensor = torch.cat(tuple(target_tensor), 0)

    n_char = input_tensor.shape[1]
    for char_idx in range(n_char):
        output, hidden = net(input_tensor[:, char_idx, :], hidden)
        # hidden = hidden.to(device)
        loss += criterion(output, target_tensor[:, char_idx])

    loss.backward()

    for p in net.parameters():
        p.data.add_(p.grad.data, alpha=-params['learning_rate'])

    return output, loss.item() / input_tensor.shape[0]

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m :2.0f}m {s :2.0f}s'

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        text = ''.join(f.readlines())

    characters = list(set(text))
    character_lookup = {character: i for i, character in enumerate(characters)}
    character_lookup["EOL"] = len(character_lookup.keys())
    n_characters = len(character_lookup.keys())
    text_length = len(text)

    os.makedirs('data/models', exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["train"]

    # if torch.cuda.is_available():  
    #     device = torch.device("cuda") 
    # else:  
    #     device = torch.device("cpu") 
    device = torch.device("cpu") 

    net = RNN(n_characters, params['hidden_size'], n_characters)

    net.to(device)

    if len(sys.argv)>2:
        net.load_state_dict(torch.load(sys.argv[2]))

    avg_loss = 0
    losses = []
    ts = time.time()
    for iter in range(params['iterations']):
        batch = get_batch(params['batch_size'], text, text_length, params['segment_length'])
        output, loss = train(net, batch, device, params)
        avg_loss += loss

        if iter % params['report_every'] == 0:
            avg_loss /= params['report_every']

            print(f"{time_since(ts)} - iteration {iter :7d} - loss {avg_loss :.4f}")
            
            for i in range(3):
                print('\n')
                print(sample(net, device, random.choice(characters), character_lookup, 1))
        
            losses.append((iter, avg_loss))
            torch.save(net.state_dict(), 'data/models/model.pth')

            

    

    with open('scores.json', "w") as fd:
        json.dump({"train_loss": avg_loss}, fd, indent=4)

    with open('loss_curve.json', "w") as fd:
        json.dump(
            {
                "prc": [
                    {"epoch": e, "loss": loss}
                    for e, loss in losses
                ]
            },
            fd,
            indent=4,
        )
