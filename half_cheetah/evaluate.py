"""evaluate.py — Half Cheetah SAC
Pulls pretrained actor from HuggingFace and evaluates over 100 episodes.
Model: https://huggingface.co/Nharen/Reward_Rush_SAC_Half_Cheetah

Requires MuJoCo: https://mujoco.readthedocs.io/en/stable/python.html
"""
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from huggingface_hub import hf_hub_download


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean    = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        return torch.tanh(self.mean(self.net(obs)))


def evaluate(num_episodes: int = 100):
    ckpt = torch.load(
        hf_hub_download("Nharen/Reward_Rush_SAC_Half_Cheetah", "half_cheetah.pth"),
        map_location="cpu", weights_only=True,
    )
    actor = SACActor(ckpt["obs_dim"], ckpt["act_dim"], ckpt.get("hidden_dim", 256))
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    env = gym.make("HalfCheetah-v4")
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            with torch.no_grad():
                action = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"Episode {ep+1:3d} | Reward: {ep_reward:.2f}")

    env.close()
    rewards = np.array(rewards)
    print(f"\nAverage Reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Min / Max:      {rewards.min():.2f} / {rewards.max():.2f}")


if __name__ == "__main__":
    evaluate()
