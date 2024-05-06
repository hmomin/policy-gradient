import gymnasium as gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from network import Network
from torch.distributions import Normal


class A2CAgent(BaseAgent):
    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        minibatch_size: int,
        num_updates: int,
        learning_rate: float,
        gamma: float,
        device: torch.device,
    ) -> None:
        super().__init__(
            env, batch_size, minibatch_size, num_updates, learning_rate, gamma, device
        )
        self.actor_log_sigma = nn.Parameter(
            torch.zeros(env.action_space.shape, dtype=torch.float32, device=self.device)
        )
        self.actor_network = Network(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.shape[0],
            output_function=nn.Tanh(),
            device=self.device,
        )
        self.critic_network = Network(
            input_size=env.observation_space.shape[0],
            output_size=1,
            device=self.device,
        )
        self.actor_optimizer = optim.Adam(
            [*self.actor_network.parameters(), self.actor_log_sigma],
            lr=learning_rate,
        )
        self.critic_optimizer = optim.Adam(
            self.critic_network.parameters(),
            lr=learning_rate,
        )
        self.save(f"VPG_{self.env_name}_{self.update_counter}_{self.step_counter}")

    def update(self) -> None:
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_returns,
        ) = self.learning_buffer
        for _ in range(self.num_updates):
            random_indices = torch.randint(
                0, batch_states.shape[0], (self.minibatch_size,), device=self.device
            )
            minibatch_states = batch_states[random_indices, :]
            minibatch_actions = batch_actions[random_indices, :]
            minibatch_returns = batch_returns[random_indices, :]
            self.update_critic(minibatch_states, minibatch_returns)
            self.update_actor(minibatch_states, minibatch_actions, minibatch_returns)
        print(
            f"Update {self.update_counter:7d} | average return: {self.get_average_episode_return():6.2f} | average stdev: {torch.mean(torch.exp(self.actor_log_sigma)).item():7.6f}"
        )
        self.reset_learning_buffer()
        self.update_counter += 1
        self.step_counter += batch_states.shape[0]
        if self.update_counter % 10 == 0:
            self.save(f"A2C_{self.env_name}_{self.update_counter}_{self.step_counter}")

    def update_critic(
        self, batch_states: torch.Tensor, batch_returns: torch.Tensor
    ) -> None:
        critic_loss = torch.nn.MSELoss()
        value_estimates = self.critic_network.forward(batch_states)
        loss = critic_loss(value_estimates, batch_returns)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_actor(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_returns: torch.Tensor,
    ) -> None:
        log_probs = self.get_log_probs(batch_states, batch_actions)
        advantages = self.compute_advantages(batch_states, batch_returns)
        actor_loss = torch.neg(torch.sum(log_probs * advantages.squeeze()))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def get_log_probs(
        self, batch_states: torch.Tensor, batch_actions: torch.Tensor
    ) -> torch.Tensor:
        means = self.actor_network.forward(batch_states)
        action_distribution = Normal(means, torch.exp(self.actor_log_sigma))
        log_probs = action_distribution.log_prob(batch_actions)
        joint_log_probs = torch.sum(log_probs, dim=1)
        return joint_log_probs

    def compute_advantages(
        self, batch_states: torch.Tensor, batch_returns: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            value_estimates = self.critic_network.forward(batch_states)
        advantages = batch_returns - value_estimates
        return advantages

    def save(self, path: str) -> None:
        if not os.path.exists("models"):
            os.mkdir("models")
        save_path = os.path.join("models", f"{path}.pt")
        torch.save(
            {
                "actor_network": self.actor_network.state_dict(),
                "actor_log_sigma": self.actor_log_sigma,
            },
            save_path,
        )
