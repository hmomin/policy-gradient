"""
Trains an Agent With Stochastic Policy Gradient Ascent to Solve Mujoco Environments
in OpenAI's Gymnasium
"""

import argparse
import gymnasium as gym
import torch
from vpg import VPGAgent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_name", help="agent to instantiate", type=str, default="VPGAgent"
    )
    parser.add_argument(
        "--env_name", help="environment to instantiate", type=str, default="Hopper-v5"
    )
    parser.add_argument(
        "--collection_steps",
        help="number of steps to run environment before doing an update",
        type=int,
        default=2_046,
    )
    parser.add_argument(
        "--total_steps",
        help="total number of steps to run environment",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate used to train all agent parameters (to keep things simple)",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--gamma",
        help="discount factor for reinforcement learning",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--render",
        help="whether or not to render the current environment",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "human" if args.render else None
    env = gym.make(args.env_name, render_mode=render_mode)

    agent = VPGAgent(
        env, learning_rate=args.learning_rate, gamma=args.gamma, device=device
    )

    state, _ = env.reset()
    batch_counter = 0
    for _ in range(args.total_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition((state, action, reward, next_state))
        batch_counter += 1

        if terminated or truncated:
            agent.handle_episode_ending(terminated, truncated)
            if batch_counter >= args.collection_steps:
                agent.update()
                batch_counter = 0
            state, _ = env.reset()
        else:
            state = next_state
    env.close()


if __name__ == "__main__":
    main()
