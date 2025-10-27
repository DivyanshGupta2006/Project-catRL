import torch
import torch.nn as nn
from torch.distributions import Normal

from src.utils import get_config

config = get_config.read_yaml()
num_assets = len(config['data']['symbols']) + 1


class Model(nn.Module):
    def __init__(self,
                 input_dim,
                 n_assets=num_assets):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = config['hyperparameters']['num_lstm_layers']
        self.n_assets = n_assets
        self.n_lstm_layers = config['hyperparameters']['num_lstm_layers']

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_layers,
            batch_first=True)

        self.actor_head = nn.Linear(self.lstm_hidden_dim, n_assets * 2)
        self.critic_head = nn.Linear(self.lstm_hidden_dim, 1)

    def init_hidden_state(self, batch_size=1, device='cpu'):
        h_0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_dim).to(device)
        c_0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_dim).to(device)
        return (h_0, c_0)

    def forward(self,
                x,
                hidden_state=None):
        lstm_out, hidden_state_out = self.lstm(x, hidden_state)
        last_out = lstm_out[:, -1, :]

        value = self.critic_head(last_out)
        actor_out = self.actor_head(last_out)

        means, log_std = torch.chunk(actor_out, 2, dim=-1)

        log_std = torch.clamp(log_std, -20, 4)
        stds = torch.exp(log_std)
        dist = Normal(means, stds)

        return dist, value, hidden_state_out

    def get_action_and_value(self,
                             x,
                             hidden_state=None,
                             action=None):
        dist, value, hidden_state_out = self.forward(x, hidden_state)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value, hidden_state_out
