import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        return self.max_action * torch.tanh(self.l3(a))

    def act(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)

        return self.max_action * torch.tanh(a), a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class LA3P_TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            device,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=0.4,
            min_priority=1
    ):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.min_priority = min_priority

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        return self.actor(state).cpu().data.numpy().flatten()

    def compute_TD_error(self, state, action, reward, next_state, not_done):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        reward = torch.FloatTensor(np.array([reward]).reshape(1, -1)).to(self.device)
        next_state = torch.FloatTensor(next_state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            TD_error_1 = (current_Q1 - target_Q).abs()
            TD_error_2 = (current_Q2 - target_Q).abs()

            Q_value = self.critic.Q1(state, action).cpu().data.numpy().flatten()
            TD_error = torch.max(TD_error_1, TD_error_2).cpu().data.numpy().flatten()

        return Q_value, TD_error

    def train_critic(self, batch_of_transitions, uniform=False):
        state, action, next_state, reward, not_done, ind, _ = batch_of_transitions

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        TD_loss_1 = (current_Q1 - target_Q)
        TD_loss_2 = (current_Q2 - target_Q)

        TD_error_1 = (current_Q1 - target_Q).abs()
        TD_error_2 = (current_Q2 - target_Q).abs()

        # Compute critic loss
        if uniform:
            critic_loss = self.PAL(TD_loss_1) + self.PAL(TD_loss_2)
            critic_loss /= torch.max(TD_loss_1.abs(), TD_loss_2.abs()).clamp(min=self.min_priority).pow(self.alpha).mean().detach()
        else:
            critic_loss = self.huber(TD_error_1) + self.huber(TD_error_2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        priority = torch.max(TD_error_1, TD_error_2).clamp(min=self.min_priority).pow(self.alpha).cpu().data.numpy().flatten()

        return ind, priority

    def train_actor(self, batch_of_transitions):
        state, _, _, _, _, _, _ = batch_of_transitions

        new_Q_value = self.critic.Q1(state, self.actor(state))
        actor_loss = -new_Q_value.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_parameters(self, replay_buffer, prioritized_fraction=0.5, batch_size=256):
        self.total_it += 1

        actor_update = self.total_it % self.policy_freq == 0

        ######################### UNIFORM SAMPLING #########################
        # Uniformly sample batch of transitions of size batch size with fraction
        if prioritized_fraction < 1.0:
            batch_of_transitions = replay_buffer.sample_uniform(int(batch_size * (1 - prioritized_fraction)))

            # Train critic and update priorities
            ind, priority = self.train_critic(batch_of_transitions, uniform=True)

            replay_buffer.update_priority_critic(ind, priority)

            # Train actor
            # Delayed policy updates
            if actor_update:
                self.train_actor(batch_of_transitions)

        ######################### CRITIC PRIORITIZED SAMPLING #########################
        # Sample critic prioritized batch of transitions
        batch_of_transitions = replay_buffer.sample_critic(int(batch_size * prioritized_fraction))

        # Train critic and update priorities according to the prioritized sampling
        ind, priority = self.train_critic(batch_of_transitions)
        replay_buffer.update_priority_critic(ind, priority)

        ######################### ACTOR PRIORITIZED SAMPLING #########################
        if actor_update:
            # Sample actor prioritized batch of transitions
            batch_of_transitions = replay_buffer.sample_actor(int(batch_size * prioritized_fraction), self.total_it)

            # Train actor and update Q-values
            self.train_actor(batch_of_transitions)

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def PAL(self, x):
        return torch.where(x.abs() < self.min_priority, (self.min_priority ** self.alpha) * 0.5 * x.pow(2), self.min_priority * x.abs().pow(1. + self.alpha) / (1. + self.alpha)).mean()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
