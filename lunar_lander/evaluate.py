"""evaluate.py — Lunar Lander DQN
Pulls pretrained model from HuggingFace and evaluates over 100 episodes.
Model: https://huggingface.co/Nharen/Reward_Rush_DQN_Lunar_Lander

Architecture note: layers are named fc1/fc2/fc3 (not nn.Sequential).
Hidden size is 32 — using 64 or 128 will cause a size mismatch.
"""
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from huggingface_hub import hf_hub_download


class LunarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(num_episodes: int = 100):
    path = hf_hub_download(
        repo_id="Nharen/Reward_Rush_DQN_Lunar_Lander",
        filename="lunar_lander_dqn.pth",
    )
    model = LunarNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("policy_net_state_dict", ckpt))
    model.eval()

    env = gym.make("LunarLander-v3")
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    env.close()
    rewards = np.array(rewards)
    print(f"Average Reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Min / Max:      {rewards.min():.2f} / {rewards.max():.2f}")


if __name__ == "__main__":
    evaluate()
