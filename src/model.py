import torch
import torch.nn as nn


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.o2o = nn.Linear(hidden_size + output_size, output_size)
#         self.dropout = nn.Dropout(0.1)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         input_combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(input_combined)
#         output = self.i2o(input_combined)
#         output_combined = torch.cat((hidden, output), 1)
#         output = self.o2o(output_combined)
#         output = self.dropout(output)
#         output = self.softmax(output)
#         return output, hidden

#     def init_hidden(self):
#         return torch.zeros(1, self.hidden_size)

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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, state):
        output, new_state = self.lstm(input, state)
        output = self.output(output)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, new_state

    def init_state(self):
        return (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))