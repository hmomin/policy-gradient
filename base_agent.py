import gymnasium as gym
import numpy as np
import time
import torch
from copy import deepcopy
from utils import safe_concat


class BaseAgent:
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
        self.clock = time.time()
        self.env_name = env.unwrapped.spec.id
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.num_updates = num_updates
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.reset_episode_buffer()
        self.reset_learning_buffer()
        self.update_counter = 0
        self.step_counter = 0

    def reset_episode_buffer(self) -> None:
        self.episode_buffer = [
            torch.zeros(0, self.state_dim, dtype=torch.float32, device=self.device),
            torch.zeros(0, self.action_dim, dtype=torch.float32, device=self.device),
            torch.zeros(0, 1, dtype=torch.float32, device=self.device),
            torch.zeros(0, self.state_dim, dtype=torch.float32, device=self.device),
        ]

    def reset_learning_buffer(self) -> None:
        # NOTE: this function assumes that the episode buffer has been reset
        self.learning_buffer = deepcopy(self.episode_buffer)
        # include the discounted returns in the learning buffer
        self.learning_buffer.append(
            torch.zeros(0, 1, dtype=torch.float32, device=self.device)
        )
        self.episode_returns: list[float] = []

    def act(self, state: np.ndarray) -> np.ndarray:
        tensor_state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_mean = self.actor_network.forward(tensor_state)
        action_sigma = torch.exp(self.actor_log_sigma)
        action_dist = torch.distributions.Normal(action_mean, action_sigma)
        action = action_dist.sample()
        return action.cpu().numpy()

    def store_transition(
        self,
        transition: tuple[np.ndarray, np.ndarray, np.float64, np.ndarray],
    ) -> None:
        transition_tensors = (
            torch.tensor(element, dtype=torch.float32, device=self.device)
            for element in transition
        )
        safe_concat(self.episode_buffer, transition_tensors)

    def handle_episode_ending(self, terminated: bool, truncated: bool) -> None:
        if not terminated and not truncated:
            raise Exception("Episode must be terminated or truncated")
        self.store_episode_return()
        discounted_returns = self.compute_discounted_returns(terminated, truncated)
        safe_concat(self.learning_buffer, (*self.episode_buffer, discounted_returns))
        self.reset_episode_buffer()

    def store_episode_return(self) -> None:
        episode_rewards = self.episode_buffer[2]
        episode_return = torch.sum(episode_rewards).item()
        self.episode_returns.append(episode_return)

    # compute the discounted return backwards through time
    def compute_discounted_returns(
        self, terminated: bool, truncated: bool
    ) -> torch.Tensor:
        if not terminated and not truncated:
            raise Exception("Episode must be terminated or truncated")
        if terminated and truncated:
            raise Exception("Episode cannot be both terminated and truncated")
        # G_0 = r_0 + γ*r_1 + (γ**2)*r_2 + (γ**3)*r_3 + ...
        #     = r_0 + γ(r_1 + γ*r_2 + (γ**2)*r_3 + ...)
        #     = r_0 + γ*G_1
        # G_t = r_t + γ*G_{t + 1}
        episode_rewards = self.episode_buffer[2]  # (H, 1)
        discounted_returns = torch.zeros_like(
            episode_rewards, dtype=torch.float32, device=self.device
        )
        # NOTE: if we have a value network, we can use it to estimate the value of the
        # next state, which is the value of the last state in the episode buffer
        # in the case that the episode is truncated - something like:
        # value_next = critic.forward(self.episode_buffer[3][-1, :])
        value_next = 0
        for t in reversed(range(0, episode_rewards.shape[0])):
            discounted_returns[t, :] = episode_rewards[t, :] + self.gamma * value_next
            value_next = discounted_returns[t, :]
        return discounted_returns

    def update(self) -> None:
        # NOTE: this function should be implemented in the child class
        raise NotImplementedError

    def get_average_episode_return(self) -> float:
        return sum(self.episode_returns) / len(self.episode_returns)
