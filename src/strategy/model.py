import torch
import torch.nn as nn
from torch.distributions import Normal

class Model(nn.Module):
    def __init__(self,
                 n_assets,
                 input_dim,
                 lstm_hidden_dim,
                 n_lstm_layers,
                 actor_hidden_dim,
                 critic_hidden_dim):
        super(Model, self).__init__()

        self.n_assets = n_assets
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_layers,
            batch_first=True)

        self.actor_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, self.n_assets * 2))

        self.critic_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.critic_hidden_dim, 1))

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

        log_std = torch.clamp(log_std, -10, 4)
        stds = torch.exp(log_std)
        dist = Normal(means, stds)

        return dist, value
