import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make("FrozenLake-v1",is_slippery = True)

state_size = env.observation_space.n
action_size = env.action_space.n

print("Imports completed and environment initialized")

world_model = np.zeros(state_size)
visited = np.zeros(state_size)

print("World Model initialized")

def choose_action(state):

  column = state%(int(state_size**0.5))
  row = (state - column)//(int(state_size**0.5))
  potential_locations = []

  neighbours = [(0,-1),(-1,0),(0,1),(1,0)]
  for i in range(len(neighbours)):
    ni = row + neighbours[i][0]
    nj = column + neighbours[i][1]

    if 0<=ni<state_size**0.5 and 0<=nj<state_size**0.5 and world_model[ni*int(state_size**0.5)+nj] != 1:
      potential_locations.append(i)

  return potential_locations[random.randint(0,len(potential_locations)-1)]

def build_world_model():

  global world_model
  global visited

  while 0 in visited:
    stepcount = 0
    state,info = env.reset()
    visited[state] = 1
    terminated = False
    truncated = False
    while not truncated and not terminated and stepcount <=95:
      action = choose_action(state)
      new_state, reward, terminated, truncated, info = env.step(action)
      stepcount += 1
      visited[new_state] = 1
      if terminated == True and reward <= 0:
        world_model[new_state] = 1
      state = new_state

  env.close()
  print("World model built")

build_world_model()

holes = []
for i in range(len(world_model)):
  if world_model[i] == 1:
    holes.append(i)

print(world_model)

Q_table_k = np.zeros((state_size,action_size))
Q_table_main = np.zeros((state_size,action_size))

print("Q-table initialized")

gamma = 0.95
max_iteration = 2000

def get_transitions(state,action):

  neighbours = [(0,-1),(1,0),(0,1),(-1,0)]
  column = state%(int(state_size**0.5))
  row = (state - column)//(int(state_size**0.5))

  possible_transitions = list(range(4))
  possible_transitions.remove((action+2)%4)
  new_states = [(row+neighbours[i][0],column+neighbours[i][1]) for i in possible_transitions]

  for i in range(len(new_states)):
    if not (0<=new_states[i][0]<state_size**0.5) or not (0<=new_states[i][1]<state_size**0.5):
      new_states[i] = (row,column)
  new_states_converted = [i[0]*int((state_size**0.5))+i[1] for i in new_states]
  return new_states_converted

def Q_iteration():
  global Q_table_k
  global Q_table_main

  Q_table_k[15,:] = 1
  Q_table_main[15,:] = 1
  Q_table_change = []

  for i in range(max_iteration):
    for j in range(state_size-1):
      if j in holes:
        continue
      for k in range(action_size):
        new_states = get_transitions(j,k)
        Q_table_main[j,k] = 0
        for l in new_states:
          Q_table_main[j,k] += gamma*Q_table_k[l,:].max()/3
    Q_table_change.append(np.sum(abs(Q_table_main-Q_table_k)))
    Q_table_k = np.copy(Q_table_main)

  for i in holes:
    Q_table_main[i,:]=0
  print("Q_table trained")
  return Q_table_change

training_data = Q_iteration()


'''
plt.plot(range(len(training_data)),training_data)
plt.xlabel("Iteration")
plt.ylabel("SumAbsError")
plt.title("Iteration Data")
plt.show()'''

epsilon = 0

def choose_action1(state):
  if random.random() < epsilon:
    action = random.randint(0,3)
  else:
    potential_actions = []
    max_q = Q_table_main[state,:].max()
    for i in range(4):
      if Q_table_main[state,i] == max_q:
        potential_actions.append(i)
    action = potential_actions[random.randint(0,len(potential_actions)-1)]

  return action
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

    while not terminated and not truncated and step < 100:

      action = choose_action1(state)
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

testing()'''

env = gym.make("FrozenLake-v1",is_slippery = True,render_mode = "rgb_array")

env = gym.wrappers.RecordVideo(
    env, 
    video_folder="stochastic_videos_Qiter", 
    episode_trigger=lambda x: True  # Records every episode
)

state,info = env.reset()
terminated = False
truncated = False

while terminated == False and truncated == False:
  action = choose_action1(state)
  new_state,reward,terminated,truncated, info = env.step(action)
  state = new_state

env.close()