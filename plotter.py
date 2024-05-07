import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pprint import pprint


data_files = [
    # "vpg_minibatch.json",
    "vpg_fullbatch.json",
    # "a2c_minibatch.json",
    "a2c_fullbatch.json",
    "ppo_minibatch.json",
]


def get_json_data(filepath: str) -> dict[str, list[float]]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def get_pruned_dicts() -> dict[str, dict[int, list[float]]]:
    pruned_dicts: dict[str, dict[int, list[float]]] = {}
    for data_filepath in data_files:
        filepath_key = ".".join(data_filepath.split(".")[:-1])
        data_dict: dict[int, list[float]] = {}
        data = get_json_data(data_filepath)
        for key, value in data.items():
            basename = os.path.basename(key)
            split_basename = re.split(r"[_.]", basename)
            full_step = int(split_basename[3])
            data_dict[full_step] = value
        pruned_dicts[filepath_key] = data_dict
    return pruned_dicts


def plot_data(pruned_dicts: dict[str, dict[int, list[float]]]) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 7))
    for key, data_dict in pruned_dicts.items():
        steps: list[int] = []
        means: list[float] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for step in sorted(data_dict.keys()):
            values = data_dict[step]
            mean = np.mean(values)
            std = np.std(values)
            steps.append(step)
            means.append(mean)
            lower_bounds.append(mean - std)
            upper_bounds.append(mean + std)
        plt.plot(steps, means, "-", label=key)
        plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.2)
    plt.xlabel("Timestep")
    plt.ylabel("Evaluation Return")
    plt.legend(loc="best")
    plt.grid(color="gray", alpha=0.2)
    plt.show()


def main() -> None:
    pruned_dicts = get_pruned_dicts()
    plot_data(pruned_dicts)


if __name__ == "__main__":
    main()
