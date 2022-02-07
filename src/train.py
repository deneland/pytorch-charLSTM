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


def train(net, batch, optimizer, params):
    criterion = torch.nn.NLLLoss()

    hidden = net.init_state(params["batch_size"])
    hidden = tuple([h.to(net.device) for h in hidden])
    net.zero_grad()

    loss = 0
    input_tensor, target_tensor = batch[0].to(net.device), batch[1].to(net.device)

    n_char = input_tensor.shape[1]
    for char_idx in range(n_char):
        output, hidden = net(input_tensor[:, char_idx, :], hidden)
        loss += criterion(output, target_tensor[:, char_idx])

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_tensor.shape[1]


def validate(net, batch, params):
    with torch.no_grad():
        criterion = torch.nn.NLLLoss()

        hidden = net.init_state(params["batch_size"])
        hidden = tuple([h.to(net.device) for h in hidden])

        net.zero_grad()

        loss = 0
        input_tensor, target_tensor = batch[0].to(net.device), batch[1].to(net.device)

        n_char = input_tensor.shape[1]
        for char_idx in range(n_char):
            output, hidden = net(input_tensor[:, char_idx, :], hidden)
            loss += criterion(output, target_tensor[:, char_idx])

        return output, loss.item() / input_tensor.shape[1]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m :2.0f}m {s :2.0f}s"


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train"]
    model_params = yaml.safe_load(open("params.yaml"))["model"]
    with open('vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    train_dataset, val_dataset = get_datasets(sys.argv[1], vocabulary, params)

    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params['workers'], drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params['workers'], drop_last=True
    )

    os.makedirs("data/models", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    n_characters = len(list(vocabulary.keys()))
    net = LanguageModel(n_characters, device, model_params)
    net.to(device)

    if len(sys.argv) > 2:
        net.load_state_dict(torch.load(sys.argv[2]))

    optimizer = torch.optim.SGD(net.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params['patience'], factor=0.5, min_lr=0.0001, verbose=True)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')

    ts = time.time()
    for epoch_idx in range(params["epochs"]):
        train_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            output, loss = train(net, batch, optimizer, params)
            train_loss += loss
        train_loss /= batch_idx
        train_losses.append((epoch_idx, train_loss))

        val_loss = 0
        for batch_idx, batch in enumerate(val_dataloader):
            output, loss = validate(net, batch, params)
            val_loss += loss
        val_loss /= batch_idx
        val_losses.append((epoch_idx, val_loss))

        scheduler.step(val_loss)

        print('\n***********************************************************')
        print(f"{time_since(ts)} - epoch {epoch_idx :7d} - train_loss {train_loss :.4f} - val_loss {val_loss :.4f} - lr {optimizer.param_groups[0]['lr'] :.4f}")

        torch.save(net.state_dict(), "data/models/last.pth")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), "data/models/best.pth")
            print('\t NEW BEST')


        print(
            sample(
                net,
                random.choice(list(vocabulary.keys())),
                vocabulary,
                1,
            )
        )


    with open("scores.json", "w") as fd:
        json.dump({"train_loss": train_losses, "validation_loss": val_losses}, fd, indent=4)

    with open("loss_curve.json", "w") as fd:
        json.dump(
            {"train_losses": [{"epoch": e, "train_loss": loss} for e, loss in enumerate(train_losses)], "val_losses": [{"epoch": e, "validationloss": loss} for e, loss in enumerate(val_losses)]},
            fd,
            indent=4,
        )
