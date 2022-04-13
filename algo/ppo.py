import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def clip_grad_norm_(parameters, max_norm):
    total_norm = torch.cat([p.grad.detach().view(-1) for p in parameters]).norm()
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef)
    return total_norm


class FixedNormal(Normal):
    def __init__(self, loc, scale, validate_args=False):
        self.loc, self.scale = loc, scale
        batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_outputs):
        super(DiagGaussian, self).__init__()
        self.logstd = AddBias(torch.zeros(num_outputs).fill_(-1))

    def forward(self, action_mean):
        zeros = torch.zeros_like(action_mean)
        action_std = self.logstd(zeros).clamp(-2.5).exp()
        return FixedNormal(action_mean, action_std)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(0))

    def forward(self, x):
        return x + self._bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_s_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("sigmoid"),
)
init_r_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("relu"),
)
init_t_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("tanh"),
)


class Policy(nn.Module):
    def __init__(self, controller):
        super(Policy, self).__init__()
        self.actor = controller
        self.dist = DiagGaussian(controller.action_dim)

        state_dim = controller.state_dim
        h_size = 256
        self.critic = nn.Sequential(
            init_r_(nn.Linear(state_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, 1)),
        )

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        mode = self.actor(inputs)
        dist = self.dist(mode)

        action = dist.mode() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(action)

        value = self.critic(inputs)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value = self.critic(inputs)
        mode = self.actor(inputs)
        dist = self.dist(mode)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class SoftsignPolicy(Policy):
    def __init__(self, controller):
        super(SoftsignPolicy, self).__init__(controller)

        state_dim = controller.state_dim
        h_size = 256
        self.critic = nn.Sequential(
            init_s_(nn.Linear(state_dim, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_s_(nn.Linear(h_size, 1)),
        )


class PolicySM(Policy):
    def __init__(self, controller):
        super().__init__(controller)

        state_dim = controller.state_dim
        h_size = 256
        self.critic = nn.Sequential(
            init_r_(nn.Linear(state_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_s_(nn.Linear(h_size, 1)),
        )


class SoftsignActor(nn.Module):
    """Simple neural net actor that takes observation as input and outputs torques"""

    def __init__(self, env, state_size=None, action_size=None):
        super().__init__()
        self.state_dim = state_size or env.observation_space.shape[0]
        self.action_dim = action_size or env.action_space.shape[0]

        h_size = 256
        self.net = nn.Sequential(
            init_s_(nn.Linear(self.state_dim, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_s_(nn.Linear(h_size, h_size)),
            nn.Softsign(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            nn.Linear(h_size, self.action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


class SoftsignActorSM(nn.Module):
    """Simple neural net actor that takes observation as input and outputs torques"""

    def __init__(self, env, state_size=None, action_size=None):
        super().__init__()
        self.state_dim = state_size or env.observation_space.shape[0]
        self.action_dim = action_size or env.action_space.shape[0]

        h_size = 256
        self.net = nn.Sequential(
            init_r_(nn.Linear(self.state_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            nn.Linear(h_size, self.action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


class PPO(object):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
    ):
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            actor_critic.parameters(),
            lr=lr,
            eps=eps,
        )

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        device = advantages.device
        value_loss_epoch = torch.tensor(0.0).to(device)
        action_loss_epoch = torch.tensor(0.0).to(device)
        dist_entropy_epoch = torch.tensor(0.0).to(device)

        clip_param = self.clip_param
        num_mini_batch = self.num_mini_batch

        for epoch in range(self.ppo_epoch):
            data_generator = rollouts.get_generator(advantages, num_mini_batch)

            for batch_i, sample in enumerate(data_generator):
                (
                    observation_batch,
                    next_obs_batch,
                    action_batch,
                    value_pred_batch,
                    return_batch,
                    old_action_log_prob_batch,
                    advantage_batch,
                ) = sample

                ac_tuples = self.actor_critic.evaluate_actions(
                    observation_batch, action_batch
                )

                values = ac_tuples[0]
                action_log_prob = ac_tuples[1]
                dist_entropy = ac_tuples[2]

                ratio = (action_log_prob - old_action_log_prob_batch).exp()
                surr1 = ratio * advantage_batch
                surr2 = (
                    ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * advantage_batch
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_pred_batch + (
                        values - value_pred_batch
                    ).clamp(-clip_param, clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad(set_to_none=True)
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()

                parameters = [
                    p for p in self.actor_critic.parameters() if p.grad is not None
                ]
                clip_grad_norm_(parameters, self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch.add_(value_loss.detach())
                action_loss_epoch.add_(action_loss.detach())
                dist_entropy_epoch.add_(dist_entropy.detach())

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch.div_(num_updates)
        action_loss_epoch.div_(num_updates)
        dist_entropy_epoch.div_(num_updates)

        return (
            value_loss_epoch.item(),
            action_loss_epoch.item(),
            dist_entropy_epoch.item(),
        )


class PPOReplayBuffer(object):
    def __init__(
        self, num_steps, num_processes, obs_dim, action_dim, device, mirror=None
    ):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_dim)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_dim)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_processes = num_processes
        self.num_steps = num_steps
        self.step = 0

        self.observations = self.observations.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

        self.mirror = mirror is not None
        if self.mirror:
            self.neg_obs_indices = torch.from_numpy(mirror[0]).to(device)
            right_obs_indices = torch.from_numpy(mirror[1]).to(device)
            left_obs_indices = torch.from_numpy(mirror[2]).to(device)
            self.neg_act_indices = torch.from_numpy(mirror[3]).to(device)
            right_action_indices = torch.from_numpy(mirror[4]).to(device)
            left_action_indices = torch.from_numpy(mirror[5]).to(device)
            self.orl = torch.cat((right_obs_indices, left_obs_indices))
            self.olr = torch.cat((left_obs_indices, right_obs_indices))
            self.arl = torch.cat((right_action_indices, left_action_indices))
            self.alr = torch.cat((left_action_indices, right_action_indices))

    def insert(
        self, current_obs, action, action_log_prob, value_pred, reward, mask, bad_mask
    ):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.bad_masks[self.step + 1].copy_(bad_mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, gamma=0.99, gae_lambda=0.95):
        self.value_preds[-1] = next_value
        scaled_deltas = self.bad_masks[1:] * (
            self.rewards
            + gamma * self.value_preds[1:] * self.masks[1:]
            - self.value_preds[:-1]
        )
        scaled_masks = gamma * gae_lambda * self.masks[1:] * self.bad_masks[1:]
        gae = 0
        for step in reversed(range(self.num_steps)):
            gae = scaled_deltas[step] + scaled_masks[step] * gae
            self.returns[step] = gae + self.value_preds[step]

    def get_generator(self, advantages, num_mini_batch):

        batch_size = self.num_processes * self.num_steps
        mini_batch_size = batch_size // num_mini_batch
        N = mini_batch_size * num_mini_batch

        device = self.rewards.device
        shuffled_indices = torch.randperm(N, generator=None, device=device)
        shuffled_indices_batch = shuffled_indices.view(num_mini_batch, -1)

        observation_shaped = self.observations.flatten(0, 1)
        action_shaped = self.actions.flatten(0, 1)
        value_pred_shaped = self.value_preds.view(-1, 1)
        return_shaped = self.returns.view(-1, 1)
        action_log_prob_shaped = self.action_log_probs.view(-1, 1)
        advantage_shaped = advantages.view(-1, 1)

        if self.mirror:
            observation_mirror_shaped = observation_shaped.clone()
            observation_mirror_shaped[:, self.neg_obs_indices] *= -1
            observation_mirror_shaped[:, self.orl] = observation_mirror_shaped[
                :, self.olr
            ]

            action_mirror_shaped = action_shaped.clone()
            action_mirror_shaped[:, self.neg_act_indices] *= -1
            action_mirror_shaped[:, self.arl] = action_mirror_shaped[:, self.alr]

        for ind in shuffled_indices_batch:

            observation_batch = observation_shaped[ind]
            next_obs_batch = observation_shaped[ind + 1]
            action_batch = action_shaped[ind]
            value_pred_batch = value_pred_shaped[ind]
            return_batch = return_shaped[ind]
            action_log_prob_batch = action_log_prob_shaped[ind]
            advantage_batch = advantage_shaped[ind]

            if self.mirror:
                observation_mirror = observation_mirror_shaped[ind]
                next_obs_mirror = observation_mirror_shaped[ind + 1]
                action_mirror = action_mirror_shaped[ind]

                observation_batch = torch.cat([observation_batch, observation_mirror])
                next_obs_batch = torch.cat([next_obs_batch, next_obs_mirror])
                action_batch = torch.cat([action_batch, action_mirror])
                value_pred_batch = value_pred_batch.repeat((2, 1))
                return_batch = return_batch.repeat((2, 1))
                action_log_prob_batch = action_log_prob_batch.repeat((2, 1))
                advantage_batch = advantage_batch.repeat((2, 1))

            yield (
                observation_batch,
                next_obs_batch,
                action_batch,
                value_pred_batch,
                return_batch,
                action_log_prob_batch,
                advantage_batch,
            )
