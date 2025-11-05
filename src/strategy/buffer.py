import torch

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.advantages = []
        self.returns = []
        self.dones = []

    def append_data(self, state, action, log_prob, value, reward, entropy, advantage, _return, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.advantages.append(advantage)
        self.returns.append(_return)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.advantages.clear()
        self.returns.clear()
        self.dones.clear()

    def display(self):
        states = torch.tensor(self.states)
        actions = torch.tensor(self.actions)
        log_probs = torch.tensor(self.log_probs)
        values = torch.tensor(self.values)
        rewards = torch.tensor(self.rewards)
        entropies = torch.tensor(self.entropies)
        advantages = torch.tensor(self.advantages)
        returns = torch.tensor(self.returns)
        dones = torch.tensor(self.dones)

        print("states:", states, states.shape)
        print("actions:", actions, actions.shape)
        print("log_probs:", log_probs, log_probs.shape)
        print("values:", values, values.shape)
        print("rewards:", rewards, rewards.shape)
        print("entropies:", entropies, entropies.shape)
        print("advantages:", advantages, advantages.shape)
        print("returns:", returns, returns.shape)
        print("dones:", dones, dones.shape)