import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

'''seedval = 24

random.seed(seedval)'''

env = gym.make("FrozenLake-v1",is_slippery = True)

number_of_observations = env.observation_space.n
number_of_actions = env.action_space.n

print("Imports completed and environment initialized.")

Q_table = np.zeros((number_of_observations,number_of_actions))

print("Q-table initialized")

training_episodes = 20000
max_steps_per_episode = 100
epsilon = 1
epsilon_decay_rate = 0.00005
min_epsilon = 0.05
gamma = 0.99
learning_rate = 0.01

print("Hyper-parameters initialized")

def choose_action(state):
  if random.random() < epsilon:
    action = random.randint(0,3)
  else:
    potential_actions = []
    max_q = Q_table[state,:].max()
    for i in range(4):
      if Q_table[state,i] == max_q:
        potential_actions.append(i)
    action = potential_actions[random.randint(0,len(potential_actions)-1)]

  return action

def agent_training():

  global epsilon
  rewards = []
  steps = []

  for episode in range(training_episodes):

    episodic_reward = 0
    step = 0
    state, info = env.reset()
    terminated,truncated = False, False

    while not terminated and not truncated and step < max_steps_per_episode:

      action = choose_action(state)
      new_state, reward, terminated, truncated, info = env.step(action)

      target = reward + gamma*(Q_table[new_state,:].max())
      error = target - Q_table[state,action]

      Q_table[state,action] += learning_rate*error

      state = new_state
      episodic_reward += reward
      step += 1

    epsilon = max(min_epsilon,epsilon-epsilon_decay_rate)
    rewards.append(reward)
    steps.append(step)

  return steps,rewards

steps,rewards = agent_training()

print("Training completed")

smooth_window = 100
smooth_rewards = [sum(rewards[max(0,i-smooth_window):i+1])/len(rewards[max(0,i-smooth_window):i+1]) for i in range(len(rewards))]
smooth_steps = [sum(steps[max(0,i-smooth_window):i+1])/len(steps[max(0,i-smooth_window):i+1]) for i in range(len(steps))]

with open("q-learning.pkl", "wb") as f:
    pickle.dump(Q_table, f)
'''
plt.plot(range(len(smooth_rewards)),smooth_rewards)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title("Learning-Rewards")
plt.show()

plt.plot(range(len(smooth_steps)),smooth_steps)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Learning-Steps")
plt.show()'''
'''
def testing():

  global epsilon

  epsilon = 0

  rewards = []
  steps = []

  testing_episodes = 100

  success = 0

  for episode in range(testing_episodes):

    episodic_reward = 0
    step = 0
    state, info = env.reset()
    terminated,truncated = False,False

    while not terminated and not truncated and step < max_steps_per_episode:

      action = choose_action(state)
      new_state, reward, terminated, truncated, info = env.step(action)
      state = new_state
      episodic_reward += reward
      step += 1

    rewards.append(reward)
    steps.append(step)

  print(f"Avg reward = {sum(rewards)/len(rewards)}")
  print(f"Max reward = {max(rewards)}")
  print(f"Min reward = {min(rewards)}")
  print(f"Avg steps = {sum(steps)/len(steps)}")

testing()

print(Q_table)'''
'''
env = gym.make("FrozenLake-v1",is_slippery = True,render_mode = "rgb_array")

env = gym.wrappers.RecordVideo(
    env, 
    video_folder="stochastic_videos_Q", 
    episode_trigger=lambda x: True  # Records every episode
)

state,info = env.reset()
terminated = False
truncated = False

while terminated == False and truncated == False:
  action = choose_action(state)
  new_state,reward,terminated,truncated, info = env.step(action)
  state = new_state

env.close()
'''