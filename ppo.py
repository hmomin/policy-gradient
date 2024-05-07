import gymnasium as gym
import torch
from a2c import A2CAgent


class PPOAgent(A2CAgent):
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
        self.clip_ratio = 0.2

    def update(self) -> None:
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_returns,
        ) = self.learning_buffer
        old_log_probs = self.get_log_probs(batch_states, batch_actions).detach()
        for _ in range(self.num_updates):
            random_indices = torch.randint(
                0, batch_states.shape[0], (self.minibatch_size,), device=self.device
            )
            minibatch_states = batch_states[random_indices, :]
            minibatch_actions = batch_actions[random_indices, :]
            minibatch_returns = batch_returns[random_indices, :]
            minibatch_old_log_probs = old_log_probs[random_indices]
            self.update_critic(minibatch_states, minibatch_returns)
            self.update_actor(
                minibatch_states,
                minibatch_actions,
                minibatch_returns,
                minibatch_old_log_probs,
            )
        print(
            f"Update {self.update_counter:7d} | Step {self.step_counter:8d} | average return: {self.get_average_episode_return():6.2f} | average stdev: {torch.mean(torch.exp(self.actor_log_sigma)).item():7.6f}"
        )
        self.update_counter += 1
        self.step_counter += batch_states.shape[0]
        if self.update_counter % 10 == 0:
            self.save(
                f"{self.__class__.__name__}_{self.env_name}_{self.update_counter}_{self.step_counter}"
            )
        self.reset_learning_buffer()

    def update_actor(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> None:
        log_probs = self.get_log_probs(states, actions)
        likelihood_ratios = torch.exp(log_probs - old_log_probs)
        advantages = self.compute_advantages(states, returns).squeeze()
        first_term = likelihood_ratios * advantages
        second_term = (
            torch.clamp(likelihood_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            * advantages
        )
        actor_loss = -torch.mean(torch.min(first_term, second_term))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
