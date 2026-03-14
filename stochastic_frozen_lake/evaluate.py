"""evaluate.py — Stochastic Frozen Lake Q-Learning
Pulls pretrained Q-table from HuggingFace and evaluates over 100 episodes.
Model: https://huggingface.co/Nharen/Reward_Rush_Q-learning_Stochastic_Frozen_Lake
"""
import gymnasium as gym
import numpy as np
import pickle
from huggingface_hub import hf_hub_download


def evaluate(num_episodes: int = 100):
    path = hf_hub_download(
        repo_id="Nharen/Reward_Rush_Q-learning_Stochastic_Frozen_Lake",
        filename="q-learning.pkl",
    )
    with open(path, "rb") as f:
        q_table = pickle.load(f)

    # Must use is_slippery=True — this agent is trained on the stochastic version
    env = gym.make("FrozenLake-v1", is_slippery=True)
    successes = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward == 1.0:
                successes += 1

    env.close()
    print(f"Goal Reach Rate: {successes}/{num_episodes}")


if __name__ == "__main__":
    evaluate()
