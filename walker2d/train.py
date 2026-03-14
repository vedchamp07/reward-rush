"""
Walker2D SAC Training Notebook
Trains a Soft Actor-Critic agent on the Walker2D environment with comprehensive tracking and visualization
Optimized for maximum performance (target: 4000-6000+ reward)
"""

# %% [markdown]
# # Walker2D SAC Training
# This notebook implements Soft Actor-Critic (SAC) to solve the Walker2D environment with comprehensive tracking and visualization.
# Target performance: 4000-5500+ reward (Good to SOTA level)

# %% Install dependencies
!apt-get install -y xvfb > /dev/null 2>&1
!pip install gymnasium[mujoco] moviepy imageio imageio-ffmpeg pyvirtualdisplay -q

# %% Setup Virtual Display
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
print("✓ Virtual display started - video recording enabled!")

# %% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import json
import os
from datetime import datetime
import imageio
import zipfile

# %% Setup paths for Kaggle
BASE_PATH = "/kaggle/working"
MODEL_PATH = os.path.join(BASE_PATH, "models")
VIDEO_PATH = os.path.join(BASE_PATH, "videos")
PLOT_PATH = os.path.join(BASE_PATH, "plots")
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(VIDEO_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# %% Actor Network (Optimized for Walker2D)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Deeper network for better performance
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# %% Critic Network (Optimized for Walker2D)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super().__init__()
        
        # Deeper networks for better value estimation
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

# %% SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                return torch.tanh(mean).cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]
    
    def update(self, batch_size, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), self.alpha

# %% Training Statistics Tracker
class StatsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.critic_losses = []
        self.actor_losses = []
        self.alphas = []
        self.eval_rewards = []
        self.best_reward = -float('inf')
        self.moving_avg_rewards = []
    
    def update(self, episode_reward, episode_length, critic_loss, actor_loss, alpha):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)
        self.alphas.append(alpha)
        
        window = min(100, len(self.episode_rewards))
        self.moving_avg_rewards.append(np.mean(self.episode_rewards[-window:]))

# %% Plotting Functions
def plot_training_stats(stats, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    axes[0, 0].plot(stats.episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0, 0].plot(stats.moving_avg_rewards, label='Moving Avg (100 eps)', linewidth=2)
    axes[0, 0].axhline(y=4000, color='g', linestyle='--', alpha=0.5, label='Good (4000)')
    axes[0, 0].axhline(y=5500, color='r', linestyle='--', alpha=0.5, label='SOTA (5500)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(stats.episode_lengths)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(stats.critic_losses)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Critic Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(stats.actor_losses)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Actor Loss')
    axes[1, 1].grid(True)
    
    axes[2, 0].plot(stats.alphas)
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Alpha')
    axes[2, 0].set_title('Temperature Parameter (Alpha)')
    axes[2, 0].grid(True)
    
    if stats.eval_rewards:
        axes[2, 1].plot(stats.eval_rewards, marker='o')
        axes[2, 1].axhline(y=4000, color='g', linestyle='--', alpha=0.5, label='Good')
        axes[2, 1].axhline(y=5500, color='r', linestyle='--', alpha=0.5, label='SOTA')
        axes[2, 1].set_xlabel('Evaluation')
        axes[2, 1].set_ylabel('Average Reward')
        axes[2, 1].set_title('Evaluation Performance')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# %% Video Recording Function
def record_video(agent, env, filepath, num_episodes=3):
    try:
        frames = []
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                frames.append(env.render())
                action = agent.select_action(state, evaluate=True)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        
        imageio.mimsave(filepath, frames, fps=30)
        print(f"Video saved to {filepath}")
    except Exception as e:
        print(f"⚠️  Video recording skipped (no display available): {str(e)[:100]}")

# %% Evaluation Function
def evaluate_agent(agent, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward
    return total_reward / num_episodes

# %% Save/Load Functions
def save_checkpoint(agent, stats, episode, filepath):
    checkpoint = {
        'episode': episode,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
        'log_alpha': agent.log_alpha,
        'alpha_optimizer': agent.alpha_optimizer.state_dict(),
        'stats': {
            'episode_rewards': stats.episode_rewards,
            'episode_lengths': stats.episode_lengths,
            'moving_avg_rewards': stats.moving_avg_rewards,
            'best_reward': stats.best_reward
        }
    }
    torch.save(checkpoint, filepath)

def save_best_model(agent, stats, filepath):
    model_data = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'metadata': {
            'best_reward': stats.best_reward,
            'total_episodes': len(stats.episode_rewards),
            'final_avg_reward': stats.moving_avg_rewards[-1] if stats.moving_avg_rewards else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'environment': 'Walker2D-v4'
        }
    }
    torch.save(model_data, filepath)
    
    metadata_path = filepath.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_data['metadata'], f, indent=4)

# %% Main Training Loop - OPTIMIZED FOR MAXIMUM PERFORMANCE
def train_sac():
    # Hyperparameters optimized for Walker2D maximum performance
    NUM_EPISODES = 4000        # Extended for Walker2D convergence
    BATCH_SIZE = 256           # Good balance for GPU
    BUFFER_SIZE = 1000000      # Large buffer for experience diversity
    WARMUP_STEPS = 10000       # Extended warmup for better exploration
    EVAL_FREQ = 50             # Regular evaluation
    SAVE_FREQ = 100            # Regular checkpoints
    UPDATES_PER_STEP = 1       # Update frequency
    
    print("=" * 70)
    print("WALKER2D SAC TRAINING - MAXIMUM PERFORMANCE CONFIGURATION")
    print("=" * 70)
    print(f"Target Performance: 4000-5500+ reward")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Warmup Steps: {WARMUP_STEPS}")
    print(f"Network: 3-layer, 400 hidden units")
    print(f"Expected Training Time: ~4-6 hours on GPU")
    print("=" * 70)
    print()
    
    # Environment
    env = gym.make('Walker2d-v4')
    eval_env = gym.make('Walker2d-v4')
    render_env = gym.make('Walker2d-v4', render_mode='rgb_array')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print()
    
    # Initialize
    agent = SAC(state_dim, action_dim)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    stats = StatsTracker()
    
    # Warmup
    print("Warming up replay buffer...")
    state, _ = env.reset()
    for step in range(WARMUP_STEPS):
        if step % 2000 == 0:
            print(f"  Warmup progress: {step}/{WARMUP_STEPS}")
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
    
    print(f"✓ Warmup complete! Buffer size: {len(replay_buffer)}\n")
    print("Starting training...")
    total_steps = 0
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        critic_losses = []
        actor_losses = []
        alphas = []
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Perform updates
            for _ in range(UPDATES_PER_STEP):
                critic_loss, actor_loss, alpha = agent.update(BATCH_SIZE, replay_buffer)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                alphas.append(alpha)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
        
        stats.update(episode_reward, episode_length, 
                    np.mean(critic_losses), np.mean(actor_losses), np.mean(alphas))
        
        # Progress logging
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg: {stats.moving_avg_rewards[-1]:.2f} | "
                  f"Steps: {total_steps}")
        
        # Milestone celebrations
        if stats.moving_avg_rewards[-1] >= 4000 and len([r for r in stats.moving_avg_rewards if r >= 4000]) == 1:
            print(f"\n🎉 MILESTONE: Reached 'Good' performance level (4000+)! 🎉\n")
        
        if stats.moving_avg_rewards[-1] >= 5500 and len([r for r in stats.moving_avg_rewards if r >= 5500]) == 1:
            print(f"\n🏆 MILESTONE: Reached SOTA performance level (5500+)! 🏆\n")
        
        # Evaluation
        if (episode + 1) % EVAL_FREQ == 0:
            eval_reward = evaluate_agent(agent, eval_env)
            stats.eval_rewards.append(eval_reward)
            print(f"📊 Evaluation at episode {episode+1}: {eval_reward:.2f}")
            
            if eval_reward > stats.best_reward:
                stats.best_reward = eval_reward
                save_best_model(agent, stats, 
                              os.path.join(MODEL_PATH, 'best_model.pt'))
                print(f"💾 New best model saved! Reward: {eval_reward:.2f}")
        
        # Save checkpoint
        if (episode + 1) % SAVE_FREQ == 0:
            checkpoint_file = os.path.join(CHECKPOINT_PATH, f'checkpoint_ep{episode+1}.pt')
            save_checkpoint(agent, stats, episode + 1, checkpoint_file)
            plot_training_stats(stats, os.path.join(PLOT_PATH, f'training_stats_ep{episode+1}.png'))
            print(f"💾 Checkpoint saved at episode {episode+1}")
    
    # Final saves
    print("\n" + "=" * 70)
    print("Training complete! Saving final results...")
    print("=" * 70)
    
    save_checkpoint(agent, stats, NUM_EPISODES, 
                   os.path.join(CHECKPOINT_PATH, 'final_checkpoint.pt'))
    plot_training_stats(stats, os.path.join(PLOT_PATH, 'final_training_stats.png'))
    
    # Record videos
    print("\nRecording videos...")
    record_video(agent, render_env, os.path.join(VIDEO_PATH, 'trained_agent.mp4'))
    
    env.close()
    eval_env.close()
    render_env.close()
    
    print(f"\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print("=" * 70)
    print(f"Best reward: {stats.best_reward:.2f}")
    print(f"Final average reward: {stats.moving_avg_rewards[-1]:.2f}")
    
    if stats.best_reward >= 5500:
        print(f"🏆 Performance Level: SOTA (Excellent!)")
    elif stats.best_reward >= 4000:
        print(f"🎉 Performance Level: Good (Well-trained!)")
    elif stats.best_reward >= 2500:
        print(f"✓ Performance Level: Decent (Functional agent)")
    else:
        print(f"⚠️  Performance Level: Below target (Consider more training)")
    print("=" * 70)
    
    return agent, stats

# %% Evaluate Best Model and Export
def evaluate_and_export_best_model():
    """Evaluate the best model and copy to Kaggle output directory"""
    
    OUTPUT_PATH = "/kaggle/working"
    KAGGLE_OUTPUT = "/kaggle/working"
    
    best_model_path = os.path.join(MODEL_PATH, 'best_model.pt')
    
    if not os.path.exists(best_model_path):
        print("Best model not found!")
        return
    
    print("Loading best model...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    
    env = gym.make('Walker2d-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(state_dim, action_dim)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    print("\nEvaluating best model over 20 episodes...")
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(20):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {ep+1}/20: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"{'='*60}\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(range(1, 21), episode_rewards, color='steelblue', alpha=0.7)
    axes[0].axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    axes[0].axhline(y=4000, color='g', linestyle='--', alpha=0.3, label='Good (4000)')
    axes[0].axhline(y=5500, color='orange', linestyle='--', alpha=0.3, label='SOTA (5500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Best Model Evaluation - Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(episode_rewards, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reward Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    eval_plot_path = os.path.join(PLOT_PATH, 'best_model_evaluation.png')
    plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plot saved to {eval_plot_path}")
    
    print("\nRecording evaluation video...")
    render_env = gym.make('Walker2d-v4', render_mode='rgb_array')
    eval_video_path = os.path.join(VIDEO_PATH, 'best_model_evaluation.mp4')
    record_video(agent, render_env, eval_video_path, num_episodes=3)
    render_env.close()
    
    eval_results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(mean_length),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_episodes': 20,
        'environment': 'Walker2d-v4'
    }
    
    eval_results_path = os.path.join(MODEL_PATH, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {eval_results_path}")
    
    import shutil
    
    print("\nCopying files to Kaggle output directory...")
    files_to_copy = [
        (best_model_path, 'best_model.pt'),
        (os.path.join(MODEL_PATH, 'best_model_metadata.json'), 'best_model_metadata.json'),
        (eval_results_path, 'evaluation_results.json'),
        (eval_plot_path, 'best_model_evaluation.png'),
        (os.path.join(PLOT_PATH, 'final_training_stats.png'), 'final_training_stats.png'),
        (eval_video_path, 'best_model_evaluation.mp4'),
        (os.path.join(VIDEO_PATH, 'trained_agent.mp4'), 'trained_agent.mp4')
    ]
    
    for src, dst_name in files_to_copy:
        if os.path.exists(src):
            dst = os.path.join(KAGGLE_OUTPUT, dst_name)
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {dst_name}")
        else:
            print(f"  ✗ Not found: {dst_name}")
    
    print("\n✅ All files ready for download from Kaggle output!")
    
    env.close()
    return eval_results

# %% Zip and Download
def create_download_package():
    """Create a comprehensive zip file of all outputs