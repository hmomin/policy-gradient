"""
Evaluates an Agent on Mujoco Environment in OpenAI's Gymnasium
"""

import argparse
import gymnasium as gym
import json
import multiprocessing
import os
import torch
import torch.nn as nn
from glob import glob
from network import Network
from pprint import pprint
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        help="directory full of models to evaluate",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_evals",
        help="number of episodes to evaluate the model",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--render",
        help="whether or not to render the current environment",
        action="store_true",
    )
    return parser.parse_args()


def get_env_name(model_dir: str) -> str:
    base_filename = os.path.basename(model_dir)
    env_name = base_filename.split("_")[1]
    return env_name


def load_agent(env: gym.Env, model_dir: str, device: torch.device) -> Network:
    torch_data = torch.load(model_dir, map_location=device)
    net_state_dict = torch_data["actor_network"]
    actor = Network(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        output_function=nn.Tanh(),
        device=device,
    )
    actor.load_state_dict(net_state_dict)
    return actor


def evaluate_episode(
    env_name: str,
    agent_name: str,
    device: torch.device,
    render_mode: str,
    episode_returns: list,
) -> None:
    env = gym.make(env_name, render_mode=render_mode)
    agent = load_agent(env, agent_name, device)
    episode_return = 0.0
    state, _ = env.reset()
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = agent.forward(state).cpu().detach().numpy()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward
    episode_returns.append(episode_return)
    env.close()


def main() -> None:
    args = get_args()
    print(f"Evaluating {args.model_dir}...")
    agent_names = glob(os.path.join(args.model_dir, "*.pt"))
    performance_dict: dict[str, list[float]] = {}
    for agent_name in tqdm(agent_names):
        env_name = get_env_name(agent_name)
        device = "cpu"
        render_mode = "human" if args.render else None

        manager = multiprocessing.Manager()
        episode_returns = manager.list()

        processes: list[multiprocessing.Process] = []

        for _ in range(args.num_evals):
            process = multiprocessing.Process(
                target=evaluate_episode,
                args=(env_name, agent_name, device, render_mode, episode_returns),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        performance_dict[agent_name] = list(episode_returns)
    json.dump(performance_dict, open("performance.json", "w"))


if __name__ == "__main__":
    main()
