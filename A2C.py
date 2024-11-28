import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Beta, Normal

def orthogonal_init(layer:nn.Module, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.alpha_layer = nn.Linear(64, args.action_dim)
        self.beta_layer = nn.Linear(64, args.action_dim)
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)
    
    def forward(self, x):
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        beta = torch.tanh(self.beta_layer(x))
        alpha = F.softplus(self.alpha_layer(x)) + 1.0
        beta = F.softplus(self.beta_layer(x)) + 1.0
        return alpha, beta
    
    def get_dist(self, x):
        alpha, beta = self.forward(x)
        dist = Beta(alpha, beta)
        return dist

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)

class A2C(object):
    def __init__(self, args):
        self.actor = Actor_Beta(args).cuda()
        self.lr_a = 3e-4
        self.lr_c = 3e-4
        self.max_step = args.max_step
        self.critic = Critic(args).cuda()
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, weight_decay=5e-4)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5, weight_decay=5e-4)
    
    def action(self, state):
        dist = self.actor.get_dist(state)
        a = dist.sample()
        return a
    
    def update(self, state, action, next_state, reward):
        action = torch.unsqueeze(action, dim=-1).cuda()
        reward = torch.unsqueeze(reward, dim=-1).cuda()
        dist = self.actor.get_dist(state)
        log_prob = dist.log_prob(action)

        value = self.critic(state)
        next_value = self.critic(next_state)

        with torch.no_grad():
            td_target = reward + 0.99 * next_value
        actor_loss = -(log_prob * (td_target - value.detach())).sum()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        critic_loss = F.mse_loss(td_target, value)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
    
    def lr_decay(self, steps):
        discount = (1 - 0.99 / (self.max_step - 1) * steps)
        lr_a_now = self.lr_a * discount
        lr_c_now = self.lr_c * discount
        for p in self.optim_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optim_critic.param_groups:
            p['lr'] = lr_c_now