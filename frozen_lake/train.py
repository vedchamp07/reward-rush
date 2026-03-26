import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v1',is_slippery = False)

print("Imports completed and environment setup")

number_of_episodes = 10000
max_steps_per_episode = 100
epsilon = 1
min_epsilon = 0.05
epsilon_decay_rate = 0.0001
learning_rate = 0.8
discount_factor = 0.95

print("Hyper-parameters defined")

Qtable = np.zeros((env.observation_space.n,env.action_space.n))

print("Qtable initialised")

def choose_action(state):

  if random.random() <= epsilon:
    action = random.randint(0,3)
  else:
    max_q = Qtable[state,:].max()
    optimal_actions_indices = []
    for i in range(len(Qtable[state,:])):
      if Qtable[state,i] == max_q:
        optimal_actions_indices.append(i)
    action = optimal_actions_indices[random.randint(0,len(optimal_actions_indices)-1)]

  return action

def training():

  global epsilon
  rewards = []
  steps = []

  for episode in range(number_of_episodes):

    state,info = env.reset()
    step = 0
    terminated = False
    reward_curepisode = 0

    while not terminated and step < max_steps_per_episode:

      action = choose_action(state)
      new_state, reward,terminated, truncated,info = env.step(action)
      step += 1
      reward_curepisode += reward

      max_q_new_state = Qtable[new_state,:].max()

      Qtable[state,action] += learning_rate*(reward + discount_factor*(max_q_new_state)-Qtable[state,action])

      state = new_state

    epsilon =  max(epsilon-epsilon_decay_rate,min_epsilon)
    steps.append(step)
    rewards.append(reward_curepisode)

  env.close()
  print("Training completed")
  return Qtable,rewards,steps

trained_Qtable, rewards_history, steps_history = training()

smooth_steps_history = [sum([i for i in steps_history[max(j-100,0):j+1]])/len(steps_history[max(j-100,0):j+1]) for j in range(len(steps_history))]
smooth_rewards_history = [sum([i for i in rewards_history[max(j-100,0):j+1]])/len(rewards_history[max(j-100,0):j+1]) for j in range(len(rewards_history))]

def plot_learning_analysis_graphs():

  plt.figure(figsize = (10,6))
  plt.plot(range(1,number_of_episodes+1), smooth_rewards_history)
  plt.title("Rewards History")
  plt.xlabel("Episode")
  plt.ylabel("Reward")

  plt.figure(figsize = (10,6))
  plt.plot(range(1,number_of_episodes+1), smooth_steps_history)
  plt.title("Steps History")
  plt.xlabel("Episode")
  plt.ylabel("Step")

  plt.show()

#plot_learning_analysis_graphs()

epsilon = 0

print(Qtable)

