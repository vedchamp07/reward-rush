import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import collections
from collections import namedtuple

# 1. Handle custom objects in the checkpoint
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
torch.serialization.add_safe_globals([collections.deque, Transition])

# 2. MATCHED ARCHITECTURE: Your file uses layer1, layer2, layer3
class MatchedNet(nn.Module):
    def __init__(self, n_observations=4, n_actions=2):
        super(MatchedNet, self).__init__()
        # Based on your error, the model uses 128 neurons per layer
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

def test_cartpole(pth_path="model.pth", num_episodes=100):
    env = gym.make("CartPole-v1")
    model = MatchedNet()
    
    try:
        # Load the nested checkpoint
        checkpoint = torch.load(pth_path, weights_only=False, map_location='cpu')
        
        # Extract weights from the key identified in previous run
        state_dict = checkpoint["policy_net_state_dict"]
        
        # Load into the matched network
        model.load_state_dict(state_dict)
        print("Successfully loaded model weights.")
    except Exception as e:
        print(f"Loading Error: {e}")
        return

    model.eval()
    rewards = []

    print(f"Testing {num_episodes} episodes (no visualization)...")
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t).max(1)[1].view(1, 1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)

    print("\n" + "="*30)
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Max/Min Score:  {np.max(rewards)}/{np.min(rewards)}")
    print("="*30)
    env.close()

if __name__ == "__main__":
    test_cartpole("model.pth")
'''

env = gym.make("CartPole-v1",render_mode = "rgb_array")
env = gym.wrappers.RecordVideo(
    env, 
    video_folder="Cartpole Video", 
    episode_trigger=lambda x: True 
)

model = MatchedNet()
checkpoint = torch.load("model.pth", weights_only=False, map_location='cpu')
model.load_state_dict(checkpoint["policy_net_state_dict"])
model.eval()
state, _ = env.reset()
episode_reward = 0
done = False
while not done:
    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state_t).max(1)[1].view(1, 1).item()
    state, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    done = terminated or truncated

env.close()'''