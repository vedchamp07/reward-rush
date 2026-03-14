"""evaluate.py — CartPole DQN
Pulls pretrained model from HuggingFace and evaluates over 100 episodes.
Model: https://huggingface.co/Nharen/Reward_Rush_DQN_Cart_Pole

Architecture note: layers are named layer1/layer2/layer3 (not fc1 or nn.Sequential).
"""
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from huggingface_hub import hf_hub_download


class MatchedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


def evaluate(num_episodes: int = 100):
    path = hf_hub_download(
        repo_id="Nharen/Reward_Rush_DQN_Cart_Pole",
        filename="Cartpole.pth",
    )
    model = MatchedNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("policy_net_state_dict", ckpt))
    model.eval()

    env = gym.make("CartPole-v1")
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
