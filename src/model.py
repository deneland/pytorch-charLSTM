import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(LSTM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.forget_gate = nn.Linear(embedding_size + hidden_size, hidden_size)

        self.input_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(embedding_size + hidden_size, hidden_size)

        self.output_gate = nn.Linear(embedding_size + hidden_size, hidden_size)

    def forward(self, input, state):
        hidden, cell_state = state
        input_combined = torch.cat((input, hidden), 1)

        forget_gate = self.sigmoid(self.forget_gate(input_combined))
        input_gate = self.sigmoid(self.input_gate(input_combined))
        update_gate = self.tanh(self.update_gate(input_combined))
        output_gate = self.sigmoid(self.output_gate(input_combined))

        updated_cell_state = (cell_state * forget_gate) + (input_gate * update_gate)

        updated_hidden = self.tanh(updated_cell_state) * output_gate

        return updated_hidden, (updated_hidden, updated_cell_state)


class LanguageModel(nn.Module):
    def __init__(self, input_size, device, params):
        super(LanguageModel, self).__init__()
        self.device = device

        self.hidden_size = params['hidden_size']

        self.lstms = nn.ModuleList()
        self.lstms.append(LSTM(input_size, self.hidden_size))#.to(device))
        for _ in range(1, params['n_layers']):
            self.lstms.append(LSTM(self.hidden_size, self.hidden_size))#.to(device))
        
        self.output = nn.Linear(self.hidden_size, input_size)
        self.dropout = nn.Dropout(params['dropout'])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, state):
        output = input

        for layer in self.lstms:
            output, state = layer(output, state)
            output = self.dropout(output)

        output = self.output(output)
        output = self.softmax(output)

        return output, state

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_size),
            torch.zeros(batch_size, self.hidden_size),
        )
