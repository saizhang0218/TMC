import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        #print('input shape is: ' + str(input_shape) + str(args.rnn_hidden_dim) + '\n\n\n\n\n\n\n\n')   #168, 64
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        #print('sss is: ' + str(inputs.shape))

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class RNNAgent_action_decoder(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent_action_decoder, self).__init__()
        self.args = args
        #print('input shape is: ' + str(input_shape) + str(args.rnn_hidden_dim) + '\n\n\n\n\n\n\n\n')   #168, 64
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_action_decoder_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_action_decoder_dim, args.rnn_hidden_action_decoder_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_action_decoder_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_action_decoder_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_action_decoder_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h