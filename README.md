[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ikU1ofUk)
<br />
<p align="center">
  <h1 align="center">Gaussian Process Proximal Policy Optimization</h1>

  <p align="center">
  </p>
</p>

## About The Project
This work introduces a new scalable model-free actor-critic based algorithm based on Proximal Policy Optimization that uses a deep Gaussian process to directly approximate both the policy and the value function. 

## Getting started

### Prerequisites
- [Poetry](https://python-poetry.org/).
## Running
This project uses  [Poetry](https://python-poetry.org/) for dependency management.
You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

Alternatively generate a requirements.txt:
```
poetry export -f requirements.txt --without-hashes > requirements.txt
```
and
```
python3 -m pip install -r requirements.txt
```

## Running the code

This project uses Hydra for configuration management. To run the code:

```bash
python main.py mode=train agent=gppo_walker2d  # (or other algorithms) or just `python main.py` for default configs
```

The codebase currently supports training, evaluation, and with optional tracking via Weights & Biases. Models can be saved and loaded. The results for each agent (GPPO, PPO, etc.) will be saved in the specified results directory. To reproduce results, ensure that `train.yaml` and `config.yaml` are configured as follows:

---

### train.yaml

```yaml
name: train

train: True
load_model: False
save_model: True
import_metrics: True
export_metrics: True
```

---

### config.yaml

```yaml
defaults:
  - mode: train   # Set a default mode value, it can be overridden by command-line args
  - agent: gppo_walker2d   # Replace with correct algorithm
  - _self_

num_episodes: 10000
num_bootstrap_samples: 100
num_runs: 1
results_save_path: "./results/"
environment: "Walker2d-v5"

normalize_obs: True
normalize_act: False
clip_obs: 10.0

wandb:
  project: "gppo-drl"
  entity: "ml_exp"
  use_wandb: True   # Disable if you do not want this
```


## Information on modules
* `agents` contains core RL implementations (e.g., `PPO`, `GPPO`) and a factory class for instantiating them.
* `gp` contains implementations of Deep Gaussian Processes (vanilla and Deep Sigma Point Process variant) and also GPPO specific implementation variants and objective functions.
* `hyperparam_tuning` contains a generic implementation of the Bayesian optimization algorithm and additional helper functions.
* `metrics` contains `MetricsTracker` class which can be used to aggregate metrics across runs.
* `util` contains utility functions and classes such as `replaybuffer` and `rolloutbuffer`.


