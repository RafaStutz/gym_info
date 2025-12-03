# gym_info

Information-theoretic diagnostics for Gymnasium environments.

`gym_info` helps you inspect **state and action entropies** in RL environments with minimal changes to your existing Gymnasium code. You wrap an environment once, run your usual interaction loop, and then query global or per-episode entropy metrics, tables, and simple reports.

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

# 4. High-level summary
summ = gi.summary(env)
gi.print_table(summ)         # text table
gi.plot_entropies(summ)      # matplotlib figure

```

## Per-episode Analysis

You can inspect entropies for each completed episode separately.

```bash
from gym_info import entropies_per_episode, episode_entropy_series, episode_entropy_dataframe

episode_metrics = entropies_per_episode(env)
for i, m in enumerate(episode_metrics):
    print(f"Episode {i}: H(S)={m.H_S:.3f}, H(A)={m.H_A:.3f}, H(A|S)={m.H_A_given_S:.3f}")

series = episode_entropy_series(env)
df = episode_entropy_dataframe(env)
```

## Discretization

gym_info works by discretizing continuous observations and actions into bins and then computing histogram-based entropies.

- A preset encapsulates reasonable default binning strategies for a family of environments.

- Currently, the main preset is:

  - preset="classic_control": designed for standard classic control environments such as CartPole-v1 and MountainCar-v0.

You can override discretization globally by passing obs_bins and action_bins to attach:

```bash
env = gi.attach(
    env,
    preset="classic_control",
    run_id="demo-run",
    obs_bins=20,    # here
    action_bins=5,  # here
)
```
