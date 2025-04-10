import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred


class ActorCriticWithLatency(nn.Module):
    def __init__(self, actor, critic=None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def forward(self, state, latency, detect_info=None):
        # in the case that critic is None, we will use the mean ACC as the critic value 
        action_pred = self.actor(state, latency, detect_info=detect_info)
        value_pred = self.critic(state, latency, detect_info=detect_info)
        
        return action_pred, value_pred

    def step(self, feat, latency, detect_res=None, 
             prev_action_state=None, prev_val_state=None):
        action_pred, action_state = self.actor.step(feat, latency, detect_res, prev_action_state)
        value_pred, value_state = self.critic.step(feat, latency, detect_res, prev_val_state)
        
        return action_pred, action_state, value_pred, value_state


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
