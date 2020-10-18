from .q_learner_6h_vs_8z import QLearner_6h_vs_8z
from .q_learner_3s_vs_4z import QLearner_3s_vs_4z
from .q_learner_3s_vs_5z import QLearner_3s_vs_5z

REGISTRY = {}

REGISTRY["q_learner_6h_vs_8z"] = QLearner_6h_vs_8z
REGISTRY["q_learner_3s_vs_4z"] = QLearner_3s_vs_4z
REGISTRY["q_learner_3s_vs_5z"] = QLearner_3s_vs_5z