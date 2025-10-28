import torch

class Buffer:
    def __init__(self):
        self.states = [] # done(ig?)
        self.actions = [] # done
        self.log_probs = [] # done
        self.advantages = [] # done
        self.returns = [] # done
        self.dones = [] # done
    def append_data(self,state,action,log_prob,advantage,_return,done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.advantages.append(advantage)
        self.returns.append(_return)
        self.dones.append(done)
    def display(self):
        states = torch.tensor(self.states)
        actions = torch.tensor(self.actions)
        log_probs = torch.tensor(self.log_probs)
        advantages = torch.tensor(self.advantages)
        returns = torch.tensor(self.returns)
        dones = torch.tensor(self.dones)

        print("states:", states, states.shape)
        print("actions:", actions, actions.shape)
        print("log_probs:", log_probs, log_probs.shape)
        print("advantages:", advantages, advantages.shape)
        print("returns:", returns, returns.shape)
        print("dones:", dones, dones.shape)