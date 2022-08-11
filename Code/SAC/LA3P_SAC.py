import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from utils import soft_update, hard_update, weights_init_

# Implementation of the SAC + LA3P algorithm

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)

        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias

        return mean

    def sample(self, state):
        mean = self.forward(state)

        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise

        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)

        return super(DeterministicPolicy, self).to(device)


class LA3P_SAC(object):
    def __init__(self, num_inputs, action_space, args, device, per_alpha=0.4, min_priority=1):
        # Initialize the training parameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.per_alpha = per_alpha

        # Initialize the policy-specific parameters
        self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        # Set CUDA device
        self.device = device
        self.min_priority = min_priority

        # Initialize critic networks and optimizer
        self.critic = Critic(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Initialize actor network and optimizer
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.actor = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def train_critic(self, batch_of_transitions, uniform=False):
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch, ind, _ = batch_of_transitions

        with torch.no_grad():
            # Select the target smoothing regularized action according to policy
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)

            # Compute the target Q-value
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Get the current Q-value estimates
        qf1, qf2 = self.critic(state_batch, action_batch)

        # Compute the critic loss
        TD_loss_1 = (qf1 - next_q_value)
        TD_loss_2 = (qf2 - next_q_value)

        TD_error_1 = (qf1 - next_q_value).abs()
        TD_error_2 = (qf2 - next_q_value).abs()

        # Compute critic loss
        if uniform:
            qf_loss = self.PAL(TD_loss_1) + self.PAL(TD_loss_2)
            qf_loss /= torch.max(TD_loss_1.abs(), TD_loss_2.abs()).clamp(min=self.min_priority).pow(self.per_alpha).mean().detach()
        else:
            qf_loss = self.huber(TD_error_1) + self.huber(TD_error_2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        priority = torch.max(TD_error_1, TD_error_2).clamp(min=self.min_priority).pow(self.per_alpha).cpu().data.numpy().flatten()

        return ind, priority

    def train_actor(self, batch_of_transitions):
        state_batch, _, _, _, _, _, _ = batch_of_transitions

        # Compute policy loss
        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Tune the temperature coefficient
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

    def update_parameters(self, replay_buffer, updates, prioritized_fraction=0.5, batch_size=256):
        ######################### UNIFORM SAMPLING #########################
        # Uniformly sample batch of transitions of size batch size with fraction
        if prioritized_fraction < 1.0:
            batch_of_transitions = replay_buffer.sample_uniform(int(batch_size * (1 - prioritized_fraction)))

            # Train critic and update priorities
            ind, priority = self.train_critic(batch_of_transitions, uniform=True)
            replay_buffer.update_priority_critic(ind, priority)

            # Train actor
            self.train_actor(batch_of_transitions)

            # Soft update the target critic network
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

        ######################### CRITIC PRIORITIZED SAMPLING #########################
        # Sample critic prioritized batch of transitions
        batch_of_transitions = replay_buffer.sample_critic(int(batch_size * prioritized_fraction))

        # Train critic and update priorities according to the prioritized sampling
        ind, priority = self.train_critic(batch_of_transitions)
        replay_buffer.update_priority_critic(ind, priority)

        # Soft update the target critic network
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        ######################### ACTOR PRIORITIZED SAMPLING #########################
        # Sample actor prioritized batch of transitions
        batch_of_transitions = replay_buffer.sample_actor(int(batch_size * prioritized_fraction), updates)

        # Train actor and update Q-values
        self.train_actor(batch_of_transitions)

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def PAL(self, x):
        return torch.where(x.abs() < self.min_priority, (self.min_priority ** self.per_alpha) * 0.5 * x.pow(2), self.min_priority * x.abs().pow(1. + self.per_alpha) / (1. + self.per_alpha)).mean()

    # Save the model parameters
    def save(self, file_name):
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

    # Load the model parameters
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = hard_update(self.critic)
