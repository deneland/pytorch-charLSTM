import sys
import torch
from torch.utils.data import DataLoader

import random
import os

from model import *
from data import *
from predict import *

import time
import math
import yaml
import json


def train(net, batch, params):
    criterion = torch.nn.NLLLoss()

    hidden = net.init_state(params["batch_size"]) 
    net.zero_grad()

    loss = 0
    input_tensor, target_tensor = batch[0].to(net.device), batch[1].to(net.device)

    n_char = input_tensor.shape[1]
    for char_idx in range(n_char-1):
        output, hidden = net(input_tensor[:, char_idx, :], hidden)
        loss += criterion(output, target_tensor[:, char_idx])

    loss.backward()

    for p in net.parameters():
        p.data.add_(p.grad.data, alpha=-params["learning_rate"])

    return output, loss.item() / input_tensor.shape[0]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m :2.0f}m {s :2.0f}s"


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train"]

    ds = TextDataset(sys.argv[1], params["segment_length"])
    dataloader = DataLoader(ds, batch_size=params["batch_size"], shuffle=False, num_workers=12)

    os.makedirs("data/models", exist_ok=True)

    with open("character_lookup.json", "w") as fd:
        json.dump(
            ds.character_lookup,
            fd,
            indent=4,
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    n_characters = len(list(ds.character_lookup.keys()))
    net = RNN(n_characters, params["hidden_size"], n_characters, device)
    net.to(device)

    if len(sys.argv) > 2:
        net.load_state_dict(torch.load(sys.argv[2]))

    avg_loss = 0
    losses = []
    ts = time.time()
    for iter_idx in range(params["iterations"]):
        batch = next(iter(dataloader))

        output, loss = train(net, batch, params)
        avg_loss += loss

        if iter_idx % params["report_every"] == 0:
            avg_loss /= params["report_every"]

            print(f"{time_since(ts)} - iteration {iter_idx :7d} - loss {avg_loss :.4f}")

            for i in range(3):
                print("\n")
                print(
                    sample(net, random.choice(list(ds.character_lookup.keys())), ds.character_lookup, 1)
                )

            losses.append((iter_idx, avg_loss))
            torch.save(net.state_dict(), "data/models/model.pth")

    with open("scores.json", "w") as fd:
        json.dump({"train_loss": avg_loss}, fd, indent=4)

    with open("loss_curve.json", "w") as fd:
        json.dump(
            {"prc": [{"epoch": e, "loss": loss} for e, loss in losses]},
            fd,
            indent=4,
        )


