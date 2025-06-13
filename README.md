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

## Usage
This project uses [Hydra](https://hydra.cc/) for configuration management. You can edit and use the configuration in `config.yaml` and run the program with:
```bash
python main.py 
```

## Information on modules


# Simulation Recordings