# gym_info

Information-theoretic diagnostics for Gymnasium environments.

`gym_info` helps you inspect **state and action entropies** in RL environments with minimal changes to your existing Gymnasium code. You wrap an environment once, run your usual interaction loop, and then query global or per-episode entropy metrics, tables, and simple reports.

`gym_info` is a **diagnostic tool**. It does **not** train policies or add exploration bonuses.
---

## Installation

```bash
pip install gym_info
# or, with uv:
# uv add gym_info
```

Requirements:
- Python: $\geq$ 3.10
- Gymnasium: $\geq$ 1.2.2

## Quick Start

```bash
import gymnasium as gym
import gym_info as gi

# 1. Create and wrap the environment
env = gym.make("CartPole-v1")
env = gi.attach(env, preset="classic_control", run_id="demo-run")

# 2. Run your usual interaction loop
obs, info = env.reset(seed=0)
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

# 3. Global entropies for the whole run (in bits)
metrics = gi.entropies(env)
print(metrics.H_S, metrics.H_A, metrics.H_A_given_S)

# 4. High-level summary and plot
summ = gi.summary(env)
gi.print_table(summ)         # text table
gi.plot_entropies(env)       # matplotlib figure (per-episode entropies)

```

## Per-episode Analysis

You can inspect entropies for each completed episode separately.

```bash
from gym_info import entropies_per_episode, episode_entropy_series, episode_entropy_dataframe

episode_metrics = entropies_per_episode(env)
for i, m in enumerate(episode_metrics):
    print(f"Episode {i}: H(S)={m.H_S:.3f}, H(A)={m.H_A:.3f}, H(A|S)={m.H_A_given_S:.3f}")

series = episode_entropy_series(env)      # convenient for plotting
df = episode_entropy_dataframe(env)       # convenient for analysis with pandas
```

## Discretization

`gym_info` works by discretizing continuous observations and actions into bins and then computing **histogram-based entropies**.

- A **preset** encapsulates a default binning strategy for a family of environments.
- Currently, the main preset is:

  - `preset="classic_control"`: designed for standard classic control environments such as `CartPole-v1` and `MountainCar-v0`, with low-dimensional observation spaces.

You can override discretization globally by passing `obs_bins` and `action_bins` to `attach`:

```bash
env = gi.attach(
    env,
    preset="classic_control",
    run_id="demo-run",
    obs_bins=20,
    action_bins=5,
)
```
### This library is not

- A training framework or RL algorithm implementation.
- An automatic way to add entropy bonuses or improve exploration during training.
- A general-purpose estimator for extremely high-dimensional or pixel-based observation spaces.


## References

 1.  Mutti, M. *Unsupervised Reinforcement Learning via State Entropy Maximization*. PhD Thesis, Università di Bologna, 2023. [url](https://amsdottorato.unibo.it/10588/1/mutti_mirco_tesi.pdf).

Proposes pre-training policies in unsupervised RL by maximizing the entropy of the visited state distribution. In many discrete experiments, state entropy is computed from visitation frequencies (histograms) and used as an intrinsic reward for exploration and transfer.

 2. Seo, Y., Gu, S., Liu, H., Srinivas, A., Abbeel, P., \& Lee, K. *State Entropy Maximization with Random Encoders for Efficient Exploration*. ICML, 2021. [url](https://proceedings.mlr.press/v139/seo21a/seo21a.pdf)

Introduces the RE3 method, which uses state entropy as an intrinsic reward for efficient exploration in high-dimensional observations. Entropy is estimated via a k-NN estimator in a latent space produced by a random, fixed convolutional encoder.

3.  Jain, A. K., Mazzaglia, P., Mutti, M., \& Restelli, M. *Maximum State Entropy Exploration using Predecessor and Successor Representations*. NeurIPS, 2023.
[url](https://papers.neurips.cc/paper_files/paper/2023/file/9c7900fac04a701cbed83256b76dbaa3-Paper-Conference.pdf)

Proposes $\eta\psi$-Learning, which learns exploratory policies by maximizing the entropy of the state distribution along a trajectory. To predict and maximize this visitation entropy, it combines predecessor and successor representations, providing a sample-based approximation of the visited-state distribution.

4. Xin, B., Liu, Y., \& Du, H. *Exploration Entropy for Reinforcement Learning*. Mathematical Problems in Engineering, 2020.
[url](https://onlinelibrary.wiley.com/doi/10.1155/2020/2672537)

Defines “exploration entropy” to quantify how much a policy explores in discrete MDPs. The metric is computed from action/state probability distributions (obtained via counting/frequencies) and is used to adjust the policy to balance exploration and reward.


## License

This project is licensed under the MIT License
