import torch

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.dones = []
    def append_data(self,state,action,log_prob,advantage,_return,done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.advantages.append(advantage)
        self.returns.append(_return)
        self.dones.append(done)