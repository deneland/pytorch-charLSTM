
import torch.onnx
import torch
import sys
import yaml
import json
from model import *

params = yaml.safe_load(open("params.yaml"))["train"]
model_params = yaml.safe_load(open("params.yaml"))["model"]
with open('vocabulary.json', 'r') as f:
    vocabulary = json.load(f)

device = torch.device('cpu')
n_characters = len(list(vocabulary.keys()))
net = LanguageModel(n_characters, device, model_params)
net.load_state_dict(torch.load(sys.argv[1]))
net.eval()

dummy_input = torch.randn(1, n_characters)
state = net.init_state(1)

output, state = net(dummy_input, state)

torch.onnx.export(net, (dummy_input, state), "data/models/export.onnx")