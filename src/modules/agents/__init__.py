REGISTRY = {}

from .rnn_agent import RNNAgent, RNNAgent_action_decoder
REGISTRY["rnn"] = RNNAgent
#REGISTRY["rnn_action_decoder"] = RNNAgent_action_decoder