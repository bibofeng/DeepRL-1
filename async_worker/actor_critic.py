#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

class AdvantageActorCritic:
    def __init__(self, config, actor, critic):
        self.config = config
        self.actor_opt = torch.optim.SGD(actor.parameters(), lr=0.0001)
        self.critic_opt = torch.optim.SGD(critic.parameters(), lr=0.0001)
        # self.optimizer = config.optimizer_fn(learning_network.parameters())
        self.worker_actor = config.actor_fn()
        self.worker_actor.load_state_dict(actor.state_dict())
        self.worker_critic = config.critic_fn()
        self.worker_critic.load_state_dict(critic.state_dict())
        self.task = config.task_fn()
        self.policy = config.policy_fn()
        self.actor = actor
        self.critic = critic

    def episode(self, deterministic=False):
        config = self.config
        state = self.task.reset()
        steps = 0
        total_reward = 0
        pending = []
        while not config.stop_signal.value and \
                (not config.max_episode_length or steps < config.max_episode_length):
            prob, log_prob = self.actor.predict(np.stack([state]))
            value = self.critic.predict(np.stack([state]))
            # prob, log_prob, value = self.worker_network.predict(np.stack([state]))
            action = self.policy.sample(prob.data.numpy().flatten(), deterministic)
            next_state, reward, terminal, _ = self.task.step(action)

            steps += 1
            total_reward += reward

            if deterministic:
                if terminal:
                    break
                state = next_state
                continue

            pending.append([prob, log_prob, value, action, reward])
            with config.steps_lock:
                config.total_steps.value += 1

            if terminal or len(pending) >= config.update_interval:
                policy_loss = 0
                value_loss = 0
                if terminal:
                    R = torch.FloatTensor([[0]])
                else:
                    R = self.worker_critic.predict(np.stack([next_state])).data
                GAE = torch.FloatTensor([[0]])
                for i in reversed(range(len(pending))):
                    prob, log_prob, value, action, reward = pending[i]
                    R = config.discount * R + reward
                    GAE = R - value.data
                    # if i == len(pending) - 1:
                    #     delta = reward + config.discount * R - value.data
                    # else:
                    #     delta = reward + pending[i + 1][2].data - value.data
                    # GAE = config.discount * config.gae_tau * GAE + delta
                    policy_loss += -log_prob.gather(1, Variable(torch.LongTensor([[action]]))) * Variable(GAE)
                    policy_loss += config.entropy_weight * torch.sum(torch.mul(prob, log_prob))

                    R = reward + config.discount * R
                    value_loss += 0.5 * (Variable(R) - value).pow(2)

                pending = []
                self.worker_actor.zero_grad()
                self.actor_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm(self.worker_actor.parameters(), config.gradient_clip)
                for param, worker_param in zip(
                        self.actor.parameters(), self.worker_actor.parameters()):
                    if param.grad is not None:
                        break
                    param._grad = worker_param.grad
                self.actor_opt.step()
                self.worker_actor.load_state_dict(self.actor.state_dict())
                # self.worker_network.reset(terminal)

                self.worker_critic.zero_grad()
                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm(self.worker_critic.parameters(), config.gradient_clip)
                for param, worker_param in zip(
                        self.critic.parameters(), self.worker_critic.parameters()):
                    if param.grad is not None:
                        break
                    param._grad = worker_param.grad
                self.critic_opt.step()
                self.worker_critic.load_state_dict(self.critic.state_dict())

            if terminal:
                break
            state = next_state

        return steps, total_reward