import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import math
import os

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

env = gym.make("LunarLander-v3")

print("Imports completed and Environment initialized")

# Improved hyperparameters
number_of_episodes = 10000
max_steps = 1000
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.0001
learning_rate = 0.001
discount_factor = 0.99

# Experience replay parameters
replay_buffer_size = 10000
batch_size = 256  # Larger batch for GPU
min_replay_size = 1000
learning_frequency = 4
target_update_frequency = 100

# Checkpoint parameters
save_frequency = 500  # Save every 500 episodes
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print("Hyper-parameters defined")

# Network definition using nn.Module for easier GPU handling
class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Main Q-Network
policy_net = DQNetwork().to(device)

# Target Q-Network (copy of main network)
target_net = DQNetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Adam optimizer for main network
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Vectorized experience replay buffer on GPU
buffer_states = torch.zeros((replay_buffer_size, 8), dtype=torch.float32, device=device)
buffer_actions = torch.zeros(replay_buffer_size, dtype=torch.long, device=device)
buffer_rewards = torch.zeros(replay_buffer_size, dtype=torch.float32, device=device)
buffer_next_states = torch.zeros((replay_buffer_size, 8), dtype=torch.float32, device=device)
buffer_dones = torch.zeros(replay_buffer_size, dtype=torch.bool, device=device)
buffer_position = 0
buffer_size = 0

print("Q Network, Target Network, and Adam Optimizer initialized")

# Training state for checkpointing
current_episode = 0
total_steps = 0
best_avg_reward = -float('inf')

def update_target_network():
    """Copy main network weights to target network"""
    target_net.load_state_dict(policy_net.state_dict())

def DQN(state):
    """Main Q-Network"""
    return policy_net(state)

def DQN_target(state):
    """Target Q-Network (for stable learning targets)"""
    return target_net(state)

def choose_action(state):
    """Epsilon-greedy action selection"""
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        Q_values = DQN(state_tensor)
    
    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        action = torch.argmax(Q_values).item()
    
    return action

def learn_from_replay():
    """Sample a batch from replay buffer and perform learning using Adam optimizer - VECTORIZED"""
    global buffer_size
    
    if buffer_size < min_replay_size:
        return
    
    # Sample random mini-batch (vectorized)
    indices = torch.randint(0, buffer_size, (batch_size,), device=device)
    
    states = buffer_states[indices]
    actions = buffer_actions[indices]
    rewards = buffer_rewards[indices]
    next_states = buffer_next_states[indices]
    dones = buffer_dones[indices]
    
    # Get Q-values for taken actions (vectorized)
    Q_values = DQN(states)
    Q_current = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values using target network (vectorized)
    with torch.no_grad():
        Q_next = torch.max(DQN_target(next_states), dim=1)[0]
        targets = rewards + discount_factor * Q_next * (~dones).float()
    
    # Compute loss
    loss = F.mse_loss(Q_current, targets)
    
    # Perform single optimization step
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    
    optimizer.step()

def save_checkpoint(episode, is_best=False):
    """Save training checkpoint"""
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'total_steps': total_steps,
        'best_avg_reward': best_avg_reward,
        'rewards_history': rewards,
        'steps_history': steps,
        'buffer_position': buffer_position,
        'buffer_size': buffer_size,
        'buffer_states': buffer_states,
        'buffer_actions': buffer_actions,
        'buffer_rewards': buffer_rewards,
        'buffer_next_states': buffer_next_states,
        'buffer_dones': buffer_dones,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model saved!")

def load_checkpoint(checkpoint_path):
    """Load training checkpoint and resume"""
    global epsilon, current_episode, total_steps, best_avg_reward
    global buffer_position, buffer_size
    global buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_dones
    global rewards, steps
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epsilon = checkpoint['epsilon']
    current_episode = checkpoint['episode'] + 1
    total_steps = checkpoint['total_steps']
    best_avg_reward = checkpoint['best_avg_reward']
    rewards = checkpoint['rewards_history']
    steps = checkpoint['steps_history']
    
    buffer_position = checkpoint['buffer_position']
    buffer_size = checkpoint['buffer_size']
    buffer_states = checkpoint['buffer_states']
    buffer_actions = checkpoint['buffer_actions']
    buffer_rewards = checkpoint['buffer_rewards']
    buffer_next_states = checkpoint['buffer_next_states']
    buffer_dones = checkpoint['buffer_dones']
    
    print(f"Resumed from episode {current_episode}, Epsilon: {epsilon:.4f}, Best Avg: {best_avg_reward:.2f}")

def training():
    global epsilon, current_episode, total_steps, best_avg_reward
    global buffer_position, buffer_size
    global rewards, steps
    
    rewards = []
    steps = []
    losses = []
    
    for episode in range(current_episode, number_of_episodes):
        state, info = env.reset()
        step = 0
        episode_reward = 0
        terminated = False
        
        while not terminated and step < max_steps:
            # Choose action using epsilon-greedy
            action = choose_action(state)
            
            # Take action in environment
            new_state, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_steps += 1
            episode_reward += reward
            
            # Store transition in replay buffer (on GPU)
            buffer_states[buffer_position] = torch.tensor(state, dtype=torch.float32, device=device)
            buffer_actions[buffer_position] = action
            buffer_rewards[buffer_position] = reward
            buffer_next_states[buffer_position] = torch.tensor(new_state, dtype=torch.float32, device=device)
            buffer_dones[buffer_position] = terminated
            
            buffer_position = (buffer_position + 1) % replay_buffer_size
            buffer_size = min(buffer_size + 1, replay_buffer_size)
            
            # Learn from replay buffer every learning_frequency steps
            if step % learning_frequency == 0:
                learn_from_replay()
            
            state = new_state
        
        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
        
        # Update target network periodically
        if episode % target_update_frequency == 0:
            update_target_network()
            if episode > 0:
                avg_reward = sum(rewards[-min(100, len(rewards)):]) / min(100, len(rewards))
                print(f"Episode {episode}: Target network updated | Epsilon: {epsilon:.4f} | Avg Reward (last 100): {avg_reward:.2f}")
        
        steps.append(step)
        rewards.append(episode_reward)
        current_episode = episode + 1
        
        # Check if best model
        if len(rewards) >= 100:
            avg_reward = sum(rewards[-100:]) / 100
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_checkpoint(episode, is_best=True)
        
        # Save checkpoint periodically
        if (episode + 1) % save_frequency == 0:
            save_checkpoint(episode)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards[-100:]) / 100
            avg_steps = sum(steps[-100:]) / 100
            print(f"Episode {episode + 1}/{number_of_episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.2f} | Epsilon: {epsilon:.4f} | Buffer Size: {buffer_size}")
    
    print("Training completed")
    env.close()
    return steps, rewards

# OPTIONAL: Load from checkpoint to resume training
# Uncomment one of these lines to resume:
# load_checkpoint('checkpoints/checkpoint_ep1000.pth')
# load_checkpoint('checkpoints/best_model.pth')

steps_history, rewards_history = training()

# Smoothing function for better visualization
smooth_steps_history = [sum([i for i in steps_history[max(j-100, 0):j+1]]) / len(steps_history[max(j-100, 0):j+1]) for j in range(len(steps_history))]
smooth_rewards_history = [sum([i for i in rewards_history[max(j-100, 0):j+1]]) / len(rewards_history[max(j-100, 0):j+1]) for j in range(len(rewards_history))]

def plot_learning_analysis_graphs():
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(rewards_history) + 1), rewards_history, alpha=0.3, label='Raw', color='blue')
    plt.plot(range(1, len(rewards_history) + 1), smooth_rewards_history, label='Smoothed (100 ep)', color='red', linewidth=2)
    plt.axhline(y=200, color='green', linestyle='--', label='Target (200)')
    plt.title("Rewards History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot steps
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(steps_history) + 1), steps_history, alpha=0.3, label='Raw', color='blue')
    plt.plot(range(1, len(steps_history) + 1), smooth_steps_history, label='Smoothed (100 ep)', color='red', linewidth=2)
    plt.title("Steps History")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    plt.subplot(1, 3, 3)
    epsilon_history = [max(1.0 - epsilon_decay_rate * i, min_epsilon) for i in range(len(rewards_history))]
    plt.plot(range(1, len(rewards_history) + 1), epsilon_history, color='purple', linewidth=2)
    plt.title("Epsilon Decay (Exploration Rate)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_learning_analysis_graphs()

# Print final statistics
print("\n" + "="*50)
print("TRAINING STATISTICS")
print("="*50)
print(f"Final Average Reward (last 100 episodes): {sum(rewards_history[-100:])/100:.2f}")
print(f"Final Average Steps (last 100 episodes): {sum(steps_history[-100:])/100:.2f}")
print(f"Best Episode Reward: {max(rewards_history):.2f}")
print(f"Best Average Reward: {best_avg_reward:.2f}")
print(f"Final Epsilon: {epsilon:.4f}")
print(f"Replay Buffer Size: {buffer_size}")

# Save the trained model
print("\nSaving trained model...")
torch.save({
    'policy_net_state_dict': policy_net.state_dict(),
    'target_net_state_dict': target_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epsilon': epsilon,
    'episode': current_episode
}, 'lunar_lander_dqn.pth')
print("Model saved as 'lunar_lander_dqn.pth'")