# π\* rates of the Gym — Reward Rush Competition

> Reinforcement learning across eight environments: from frozen grids to bipedal locomotion.

**Team:** Vedant Narayanswami (EP25B040) · M Nharen (EE25B089) — IIT Madras  
**Competition:** Reward Rush · January 2026  
**Website:** [π\* rates of the Gym](https://sites.google.com/view/rewardrush-teampiratesofthegym/home)  
**Report:** [`assets/report/rewardrushreportv3.pdf`](assets/report/rewardrushreportv3.pdf)

---

## Pretrained Models

All trained models are hosted on 🤗 Hugging Face. Each model card has architecture details, common pitfalls, and a working eval script.

| Environment | Algorithm | Model |
|---|---|---|
| Frozen Lake | Q-Learning | [🤗 Nharen/Reward\_Rush\_Q-learning\_Frozen\_lake\_Deterministic](https://huggingface.co/Nharen/Reward_Rush_Q-learning_Frozen_lake_Deterministic) |
| Taxi | Q-Learning | [🤗 Nharen/Reward\_Rush\_Q-learning\_Taxi](https://huggingface.co/Nharen/Reward_Rush_Q-learning_Taxi) |
| Stochastic Frozen Lake | Q-Learning | [🤗 Nharen/Reward\_Rush\_Q-learning\_Stochastic\_Frozen\_Lake](https://huggingface.co/Nharen/Reward_Rush_Q-learning_Stochastic_Frozen_Lake) |
| Lunar Lander | DQN | [🤗 Nharen/Reward\_Rush\_DQN\_Lunar\_Lander](https://huggingface.co/Nharen/Reward_Rush_DQN_Lunar_Lander) |
| CartPole | DQN | [🤗 Nharen/Reward\_Rush\_DQN\_Cart\_Pole](https://huggingface.co/Nharen/Reward_Rush_DQN_Cart_Pole) |
| Mountain Car Continuous | SAC | [🤗 Nharen/Reward\_Rush\_SAC\_Mountain\_Car](https://huggingface.co/Nharen/Reward_Rush_SAC_Mountain_Car) |
| Half Cheetah | SAC | [🤗 Nharen/Reward\_Rush\_SAC\_Half\_Cheetah](https://huggingface.co/Nharen/Reward_Rush_SAC_Half_Cheetah) |
| Walker2D | SAC | [🤗 Nharen/Reward\_Rush\_SAC\_Walker](https://huggingface.co/Nharen/Reward_Rush_SAC_Walker) |

---

## Results

| Environment | Algorithm | Avg. Return | Notes |
|---|---|---|---|
| Frozen Lake | Q-Learning | 100% success | Optimal path every episode |
| Taxi | Q-Learning | 100% success | Avg reward 7.89 |
| Stochastic Frozen Lake | Q-Learning | 75% success rate | Variance from env stochasticity |
| Lunar Lander | DQN | 260.00 | 100 episodes |
| CartPole | DQN | **500.00** | Perfect score, zero variance |
| Mountain Car Continuous | SAC | **96.07** | 100% success rate |
| Half Cheetah | SAC | **9692.19 ± 142** | Max: 9969.90 |
| Walker2D | SAC | **4290.40 ± 37.65** | Max: 4432.63 |

Training curves for each environment are in `<env>/results/`.

---

## Quickstart

```bash
pip install -r requirements.txt

# Evaluate any agent — downloads the model from HuggingFace automatically
python frozen_lake/evaluate.py
python lunar_lander/evaluate.py
python walker2d/evaluate.py

# Train from scratch
python walker2d/train.py
```

> **MuJoCo required** for Half Cheetah and Walker2D.  
> Setup guide: https://mujoco.readthedocs.io/en/stable/python.html

---

## Approach

We matched each algorithm to the structural properties of its environment.

**Tabular Q-Learning** for small, fully discrete MDPs (Frozen Lake, Taxi, Stochastic Frozen Lake). A lookup table over state-action pairs is sufficient and guaranteed to converge.

**DQN** when the state space is continuous but actions stay discrete (Lunar Lander, CartPole). A neural network approximates Q-values, trained with experience replay.

**SAC** for fully continuous control (Mountain Car, Half Cheetah, Walker2D). Entropy regularization prevents premature convergence and greatly improves sample efficiency.

---

## Project Structure

```
reward-rush/
├── frozen_lake/
│   ├── train.py
│   ├── evaluate.py       ← pulls model from HuggingFace, runs 100 episodes
│   ├── config.yaml
│   └── results/          ← training curves
├── taxi/
├── stochastic_frozen_lake/
├── lunar_lander/
├── cartpole/
├── mountain_car_continuous/
├── half_cheetah/
├── walker2d/
│   └── notebooks/
├── experiments/          ← PPO ablations (Walker2D, HalfCheetah)
├── assets/
│   └── report/           ← rewardrushreportv3.pdf + .tex source
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Key Hyperparameters

### Tabular (Frozen Lake · Taxi · Stochastic Frozen Lake)
| Param | Frozen Lake / Taxi | Stochastic FL |
|---|---|---|
| Learning rate | 0.8 | 0.01 |
| Discount γ | 0.95 | 0.99 |
| ε decay | 0.0001 | 0.00005 |
| Episodes | 10,000 | 20,000 |

### DQN (Lunar Lander · CartPole)
| Param | Lunar Lander | CartPole |
|---|---|---|
| Learning rate | 1e-3 | 1e-3 |
| Hidden layers | 2 × 32 | 2 × 128 |
| Replay buffer | 10,000 · batch 256 | 10,000 · batch 256 |
| Discount γ | 0.99 | 0.99 |

### SAC (Mountain Car · Half Cheetah · Walker2D)
| Param | Value |
|---|---|
| Learning rate | 3e-4 |
| Discount γ | 0.99 |
| Soft update τ | 0.005 |
| Entropy tuning | Automatic (init α = 0.2) |
| Replay buffer | 100K (MountainCar) / 1M (Cheetah, Walker) |

---

## Experiments

`experiments/` contains PPO runs on Half Cheetah and Walker2D conducted before settling on SAC. SAC outperformed PPO on both tasks in final return and training stability.

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290), 2018
- Mnih et al., [Human-level control through deep RL](https://www.nature.com/articles/nature14236), 2015
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
