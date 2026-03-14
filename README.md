# π\* rates of the Gym

> Eight environments. One month. First place.

**Team:** Vedant Narayanswami (EP25B040) · M Nharen (EE25B089) — IIT Madras  
**Competition:** Reward Rush · iBoT Club, IIT Madras · December 2025  
**Website:** [π\* rates of the Gym](https://sites.google.com/view/rewardrush-teampiratesofthegym/home) · **Report:** [`assets/report/rewardrushreportv3.pdf`](assets/report/rewardrushreportv3.pdf)

> 🥇 **First Place** — Reward Rush, iBoT Club, IIT Madras (December 2025)

---

## Results

| Environment             | Algorithm  | Result                         |
| ----------------------- | ---------- | ------------------------------ |
| Frozen Lake             | Q-Learning | 100% success rate              |
| Taxi                    | Q-Learning | 100% success · 7.89 avg reward |
| Stochastic Frozen Lake  | Q-Learning | 75% success rate               |
| Lunar Lander            | DQN        | 260.18 avg · max 307           |
| CartPole                | DQN        | **500.00 — perfect score**     |
| Mountain Car Continuous | SAC        | **96.07 avg · 100% success**   |
| Half Cheetah            | SAC        | **9692 avg · max 9969**        |
| Walker2D                | SAC        | **4290 avg · max 4432**        |

---

## Pretrained Models

All weights live on 🤗 Hugging Face — each model card has the architecture, common gotchas, and a working eval script.

| Environment             | Model                                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Frozen Lake             | [Nharen/Reward_Rush_Q-learning_Frozen_lake_Deterministic](https://huggingface.co/Nharen/Reward_Rush_Q-learning_Frozen_lake_Deterministic) |
| Taxi                    | [Nharen/Reward_Rush_Q-learning_Taxi](https://huggingface.co/Nharen/Reward_Rush_Q-learning_Taxi)                                           |
| Stochastic Frozen Lake  | [Nharen/Reward_Rush_Q-learning_Stochastic_Frozen_Lake](https://huggingface.co/Nharen/Reward_Rush_Q-learning_Stochastic_Frozen_Lake)       |
| Lunar Lander            | [Nharen/Reward_Rush_DQN_Lunar_Lander](https://huggingface.co/Nharen/Reward_Rush_DQN_Lunar_Lander)                                         |
| CartPole                | [Nharen/Reward_Rush_DQN_Cart_Pole](https://huggingface.co/Nharen/Reward_Rush_DQN_Cart_Pole)                                               |
| Mountain Car Continuous | [Nharen/Reward_Rush_SAC_Mountain_Car](https://huggingface.co/Nharen/Reward_Rush_SAC_Mountain_Car)                                         |
| Half Cheetah            | [Nharen/Reward_Rush_SAC_Half_Cheetah](https://huggingface.co/Nharen/Reward_Rush_SAC_Half_Cheetah)                                         |
| Walker2D                | [Nharen/Reward_Rush_SAC_Walker](https://huggingface.co/Nharen/Reward_Rush_SAC_Walker)                                                     |

---

## Quickstart

```bash
pip install -r requirements.txt

# Evaluate any agent — downloads weights from Hugging Face automatically
python frozen_lake/evaluate.py
python lunar_lander/evaluate.py
python walker2d/evaluate.py

# Train from scratch
python walker2d/train.py
```

> MuJoCo required for Half Cheetah and Walker2D. [Setup guide →](https://mujoco.readthedocs.io/en/stable/python.html)

---

## How we approached it

We spent the month matching algorithms to the structure of each environment rather than throwing the same method at everything.

**Tabular Q-Learning** for the discrete grid worlds (Frozen Lake, Taxi, Stochastic Frozen Lake). When the entire state space fits in a table, a neural network is overkill — Q-learning converges cleanly and is dead simple to debug.

**DQN** for continuous states with discrete actions (Lunar Lander, CartPole). We actually started CartPole with REINFORCE, but policy gradient variance made training painfully unstable. Switching to DQN with experience replay got us to a perfect 500.00 in a fraction of the time. The REINFORCE run lives in `experiments/cartpole_reinforce/` as a cautionary tale.

**SAC** for fully continuous control (Mountain Car, Half Cheetah, Walker2D). Entropy regularisation was the key — it keeps the policy exploring instead of collapsing early, which matters a lot on tasks as hard as Walker2D. We also ran PPO on Half Cheetah and Walker2D before settling on SAC; SAC won on both final return and training stability (see `experiments/`).

---

## Project Structure

```
reward-rush/
├── frozen_lake/             ← Q-Learning
├── taxi/                    ← Q-Learning
├── stochastic_frozen_lake/  ← Q-Learning + Q-Value Iteration
├── lunar_lander/            ← DQN
├── cartpole/                ← DQN
├── mountain_car_continuous/ ← SAC
├── half_cheetah/            ← SAC
├── walker2d/                ← SAC
│   └── notebooks/
├── experiments/             ← PPO ablations, REINFORCE attempt
├── assets/
│   └── report/
├── README.md
├── requirements.txt
└── .gitignore
```

Each environment folder follows the same layout:

```
<env>/
├── train.py      ← training loop
├── evaluate.py   ← pulls weights from HF, runs 100 episodes
└── results/      ← training curves and plots
```

---

## Hyperparameters

### Tabular

|               | Frozen Lake / Taxi | Stochastic FL |
| ------------- | ------------------ | ------------- |
| Learning rate | 0.8                | 0.01          |
| Discount γ    | 0.95               | 0.99          |
| ε decay       | 0.0001             | 0.00005       |
| Episodes      | 10,000             | 20,000        |

### DQN

|               | Lunar Lander    | CartPole        |
| ------------- | --------------- | --------------- |
| Learning rate | 1e-3            | 1e-3            |
| Hidden size   | 2 × 32          | 2 × 128         |
| Replay buffer | 10K · batch 256 | 10K · batch 256 |
| Discount γ    | 0.99            | 0.99            |

### SAC

| Param          | Value                                     |
| -------------- | ----------------------------------------- |
| Learning rate  | 3e-4                                      |
| Discount γ     | 0.99                                      |
| Soft update τ  | 0.005                                     |
| Entropy tuning | Automatic (α init 0.2)                    |
| Replay buffer  | 100K (MountainCar) · 1M (Cheetah, Walker) |

---

## References

- Sutton & Barto, _Reinforcement Learning: An Introduction_ (2nd ed.)
- Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290), 2018
- Mnih et al., [Human-level control through deep RL](https://www.nature.com/articles/nature14236), 2015
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
