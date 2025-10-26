import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_features, lstm_hidden_size, action_dim=3):
        super(ActorCriticLSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_features, lstm_hidden_size, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size // 2, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size // 2, 1)
        )

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)

        features = lstm_out[:, -1, :]
        action_logits = self.actor(features)
        distribution = Categorical(logits=action_logits)

        value = self.critic(features)
        value = value.squeeze(-1)
        return distribution, value, (h_n, c_n)
    
    def evaluate_actions(self, states, actions, hidden_state=None):
        distribution, values, final_hidden_state = self.forward(states, hidden_state)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, values, entropy, final_hidden_state