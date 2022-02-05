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

def train(net, line, device, params):
    criterion = torch.nn.NLLLoss()

    hidden = net.init_hidden().to(device)
    net.zero_grad()

    loss = 0
    input_tensor = get_input_tensor(line).to(device)
    target_tensor = get_target_tensor(line).to(device)

    for i, char_tensor in enumerate(input_tensor):
        output, hidden = net(char_tensor, hidden)
        hidden = hidden.to(device)
        loss += criterion(output[0], target_tensor[i])

        # print(output[0])
        # print(target_tensor[i])
        # print(target_tensor[i].unsqueeze(0).shape)


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
        lines = [clean_line(line) for line in f.readlines()]

    # print(lines)
    os.makedirs('data/models', exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["train"]

    # if torch.cuda.is_available():  
    #     device = torch.device("cuda") 
    # else:  
    #     device = torch.device("cpu") 
    device = torch.device("cpu") 

    net = RNN(n_characters, 128, n_characters)

    net.to(device)

    if len(sys.argv)>2:
        net.load_state_dict(torch.load(sys.argv[2]))

    iter = 0

    ts = time.time()
    for e in range(params['epochs']):
        random.shuffle(lines)
        all_losses = 0
        for i in range(len(lines)):
            iter += 1
            
            output, loss = train(net, lines[i], device, params)
            all_losses += loss
            avg_loss = all_losses/(i+1)

            if iter % 5000 == 0:
                print(f"{time_since(ts)} - iteration {iter :7d} - epoch {e+1} ({i/len(lines)*100 :3.0f}%) - loss {avg_loss :.4f}")
                for i in range(3):
                    print('\t' + sample(net, device, random.choice(characters)))

            

    torch.save(net.state_dict(), 'data/models/model.pth')

    with open('scores.json', "w") as fd:
        json.dump({"train_loss": avg_loss}, fd, indent=4)
