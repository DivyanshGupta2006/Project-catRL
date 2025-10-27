import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus, softmax

from src.utils import get_config

config = get_config.read_yaml()
num_assets = len(config['data']['symbols']) + 1


class Model(nn.Module):
    def __init__(self,
                 input_dim,
                 n_assets=num_assets):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = config['hyperparameters']['hidden_state_dim']
        self.n_assets = n_assets
        self.n_lstm_layers = config['hyperparameters']['num_lstm_layers']
        self.actor_hidden_dim = config['hyperparameters']['actor_hidden_dim']
        self.critic_hidden_dim = config['hyperparameters']['critic_hidden_dim']

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_layers,
            batch_first=True)

        self.actor_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.actor_hidden_dim),
            nn.Linear(self.actor_hidden_dim, n_assets * 2))

        self.critic_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.critic_hidden_dim),
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

    @staticmethod
    def get_fiduciae(actor_out):
        actor_out = actor_out.squeeze()

        fiduciae = torch.zeros_like(actor_out)

        # Separate crypto and cash actor_out
        crypto_actor_out = actor_out[:-1]  # First 9 elements
        cash_logit = actor_out[-1]  # Last element

        processed_cash_logit = softplus(cash_logit).unsqueeze(0)  # Shape [1]

        #Long Book
        long_mask = crypto_actor_out >= 0
        long_indices = long_mask.nonzero(as_tuple=True)[0]
        long_crypto_actor_out = crypto_actor_out[long_mask]

        long_book_actor_out = torch.cat([long_crypto_actor_out, processed_cash_logit])

        long_book_fiduciae = softmax(long_book_actor_out, dim=0)

        fiduciae[long_indices] = long_book_fiduciae[:-1]
        fiduciae[-1] = long_book_fiduciae[-1]

        #Short Book
        short_mask = crypto_actor_out < 0
        short_indices = short_mask.nonzero(as_tuple=True)[0]
        short_crypto_actor_out = crypto_actor_out[short_mask]

        if short_crypto_actor_out.numel() > 0:
            abs_short_actor_out = torch.abs(short_crypto_actor_out)

            short_book_fiduciae_abs = softmax(abs_short_actor_out, dim=0)

            fiduciae[short_indices] = -short_book_fiduciae_abs

        return fiduciae

