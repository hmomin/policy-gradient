import gymnasium as gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from network import Network
from torch.distributions import Normal


class VPGAgent(BaseAgent):
    def __init__(
        self, env: gym.Env, learning_rate: float, gamma: float, device: torch.device
    ) -> None:
        self.env_name = env.unwrapped.spec.id
        super().__init__(env, learning_rate, gamma, device)
        self.update_counter = 0
        self.step_counter = 0
        self.actor_log_sigma = nn.Parameter(
            torch.zeros(env.action_space.shape, dtype=torch.float32, device=self.device)
        )
        self.actor_network = Network(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.shape[0],
            output_function=nn.Tanh(),
            device=self.device,
        )
        self.optimizer = optim.Adam(
            [*self.actor_network.parameters(), self.actor_log_sigma],
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
        means = self.actor_network.forward(batch_states)
        action_distribution = Normal(means, torch.exp(self.actor_log_sigma))
        log_probs = action_distribution.log_prob(batch_actions)
        joint_lob_probs = torch.sum(log_probs, dim=1)
        L = torch.neg(torch.sum(joint_lob_probs * batch_returns.squeeze()))
        self.gradient_descent_step(L)
        print(
            f"Update {self.update_counter:7d} | average return: {self.get_average_episode_return():6.2f} | average stdev: {torch.mean(torch.exp(self.actor_log_sigma)).item():7.6f}"
        )
        self.reset_learning_buffer()
        self.update_counter += 1
        self.step_counter += batch_states.shape[0]
        if self.update_counter % 10 == 0:
            self.save(
                f"vpg_agent_{self.env_name}_{self.update_counter}_{self.step_counter}"
            )

    def get_average_episode_return(self) -> float:
        return sum(self.episode_returns) / len(self.episode_returns)

    def gradient_descent_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
