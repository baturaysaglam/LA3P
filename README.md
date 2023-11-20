# Actor Prioritized Experience Replay

### Published!
If you use our code or results, please cite the paper.
```
@article{Saglam2023,
  title = {Actor Prioritized Experience Replay},
  volume = {78},
  ISSN = {1076-9757},
  url = {http://dx.doi.org/10.1613/jair.1.14819},
  DOI = {10.1613/jair.1.14819},
  journal = {Journal of Artificial Intelligence Research},
  publisher = {AI Access Foundation},
  author = {Saglam,  Baturay and Mutlu,  Furkan B. and Cicek,  Dogan C. and Kozat,  Suleyman S.},
  year = {2023},
  month = nov,
  pages = {639–672}
}
```
#
PyTorch implementation of the _Loss Adjusted Approximate Actor Prioritized Experience Replay_ algorithm (LA3P). If you use our code or results, please cite the [paper](https://arxiv.org/abs/2209.00532). 
Note that the implementation of the baseline algorithms are heavily based on the following repositories:

- [SAC](https://arxiv.org/abs/1801.01290): Our implementation. Uses the precise hyper-parameter settings provided in the original article.
- [TD3](https://arxiv.org/abs/1802.09477): The fine-tuned version imported from the [author's Pytorch implementation of the TD3 algorithm](https://github.com/sfujim/TD3). 

The algorithm is tested on [MuJoCo](https://gym.openai.com/envs/#mujoco) and [Box2D](https://gym.openai.com/envs/#box2d) continuous control suites.

### Results
Learning curves are found under [./Learning Curves](https://github.com/baturaysaglam/LA3P/tree/main/Learning%20Curves). 
Corresponding learning figures are found under [./Learning Figures](https://github.com/baturaysaglam/LA3P/tree/main/Learning%20Figures). 
Each learning curve is formatted as NumPy arrays of 1001 evaluations (1001,), except for the Ant, HalfCheetah, Humanoid, and Swimmer environments, which are of 2001 evaluations (2001,). Each evaluation corresponds to the average reward from running the policy for 10 episodes without exploration and updates. 

The randomly initialized policy network produces the first evaluation. Evaluations are performed every 1000 time steps, over 1 million (2 million for Ant, HalfCheetah, Humanoid, and Swimmer) time steps for 10 random seeds.

### Computing Infrastructure
Following computing infrastructure is used to produce the results.
| Hardware/Software  | Model/Version |
| ------------- | ------------- |
| Operating System  | Ubuntu 18.04.5 LTS  |
| CPU  | AMD Ryzen 7 3700X 8-Core Processor |
| GPU  | Nvidia GeForce RTX 2070 SUPER |
| CUDA  | 11.1  |
| Python  | 3.8.5 |
| PyTorch  | 1.8.1 |
| OpenAI Gym  | 0.17.3 |
| MuJoCo  | 1.50 |
| Box2D  | 2.3.10 |
| NumPy  | 1.19.4 |

### Usage - TD3
```
usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--start_time_steps N] [--buffer_size BUFFER_SIZE]
               [--prioritized_fraction PRIORITIZED_FRACTION] [--eval_freq N]
               [--max_time_steps N] [--exploration_noise G] [--batch_size N]
               [--discount G] [--tau G] [--policy_noise G] [--noise_clip G]
               [--policy_freq N] [--save_model] [--load_model LOAD_MODEL]
```

### Arguments - TD3
```
Twin Delayed Deep Deterministic Policy Gradient

optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: LA3P_TD3)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                        (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_time_steps N  Number of exploration time steps sampling random
                        actions (default: 25000)
  --buffer_size BUFFER_SIZE
                        Size of the experience replay buffer (default:
                        1000000)
  --prioritized_fraction PRIORITIZED_FRACTION
                        Fraction of prioritized sampled batch of transitions
  --eval_freq N         Evaluation period in number of time steps (default:
                        1000)
  --max_time_steps N    Maximum number of steps (default: 1000000)
  --exploration_noise G
                        Std of Gaussian exploration noise
  --batch_size N        Batch size (default: 256)
  --discount G          Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target
                        networks (default: 0.005)
  --policy_noise G      Noise added to target policy during critic update
  --noise_clip G        Range to clip target policy noise
  --policy_freq N       Frequency of delayed policy updates
  --save_model          Save model and optimizer parameters
  --load_model LOAD_MODEL
                        Model load file name; if empty, does not load
  ```

### Usage - SAC
```
usage: main.py [-h] [--policy POLICY] [--policy_type POLICY_TYPE] [--env ENV]
               [--seed SEED] [--gpu GPU] [--start_steps N]
               [--buffer_size BUFFER_SIZE]
               [--prioritized_fraction PRIORITIZED_FRACTION] [--eval_freq N]
               [--num_steps N] [--batch_size N] [--hard_update G]
               [--train_freq N] [--updates_per_step N]
               [--target_update_interval N] [--alpha G]
               [--automatic_entropy_tuning G] [--reward_scale N] [--gamma G]
               [--tau G] [--lr G] [--hidden_size N]
```

### Arguments - SAC
```
Soft Actor-Critic

optional arguments:
  -h, --help            show this help message and exit
  --policy POLICY       Algorithm (default: LA3P_SAC)
  --policy_type POLICY_TYPE
                        Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --env ENV             OpenAI Gym environment name
  --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                        (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --start_steps N       Number of exploration time steps sampling random
                        actions (default: 25000)
  --buffer_size BUFFER_SIZE
                        Size of the experience replay buffer (default:
                        1000000)
  --prioritized_fraction PRIORITIZED_FRACTION
                        Fraction of prioritized sampled batch of transitions
  --eval_freq N         evaluation period in number of time steps (default:
                        1000)
  --num_steps N         Maximum number of steps (default: 1000000)
  --batch_size N        Batch size (default: 256)
  --hard_update G       Hard update the target networks (default: True)
  --train_freq N        Frequency of the training (default: 1)
  --updates_per_step N  Model updates per training time step (default: 1)
  --target_update_interval N
                        Number of critic function updates per training time
                        step (default: 1)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        Automatically adjust α (default: True)
  --reward_scale N      Scale of the environment rewards (default: 5)
  --gamma G             Discount factor for reward (default: 0.99)
  --tau G               Learning rate in soft/hard updates of the target
                        networks (default: 0.005)
  --lr G                Learning rate (default: 0.0003)
  --hidden_size N       Hidden unit size in neural networks (default: 256)
```
