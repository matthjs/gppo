from typing import Union
from src.agents.agentfactory import AgentFactory
from src.util.interaction import create_agent_for_catch_env
from typing import Dict, Any
from src.agents.agent import Agent
from src.util.interaction import agent_env_loop
import gymnasium as gym


def train_rl_agent(agent: Agent, params: Dict[str, Any], env: gym.Env,
                   normalize_obs: bool = True) -> Dict[str, float]:
    """
    Train the RL agent with the given hyperparameters.

    :param agent: RL Agent to train.
    :param params: The hyperparameters to use during training.
    :param env: The environment to train in (must follow Gym API).
    :return: Dictionary with training metrics.
    """
    avg_return = agent_env_loop(
        agent=agent,
        num_episodes=params['num_episodes'],
        wandb_logger=params.get("wandb_logger", None),
        learning=True,
        env=env,
        verbose=params.get("verbose", False),
        normalize_obs=normalize_obs
    )
    return {"train/avg_return": avg_return}


def eval_rl_agent(agent: Agent, params: Dict[str, Any], env: gym.Env,
                  normalize_obs: bool = True) -> Dict[str, float]:
    """
    Evaluate the RL agent (no learning updates).

    :param agent: RL Agent to evaluate.
    :param params: Hyperparameters (used for logging/evaluation).
    :param env: The environment to evaluate in.
    :return: Dictionary with evaluation metrics.
    """
    avg_return = agent_env_loop(
        agent=agent,
        num_episodes=params['num_eval_episodes'],
        wandb_logger=params.get("wandb_logger", None),
        learning=False,
        env=env,
        verbose=params.get("verbose", False),
        normalize_obs=normalize_obs
    )
    return {"return": avg_return}


def create_rl_agent(params: Dict[str, Any], env: Union[str, gym.Env]) -> Agent:
    """
    Create an RL agent using the AgentFactory with a specified environment.

    :param env: A Gym environment instance or string name.
    :param params: Dictionary of agent parameters, must include 'agent_type'.
    :return: Instantiated Agent.
    """
    param_copy = dict(params)
    agent_type = param_copy.pop('agent_type')

    # Clean up any non-agent hyperparameters to avoid constructor issues
    param_copy.pop('num_episodes', None)
    param_copy.pop('num_eval_episodes', None)
    param_copy.pop('wandb_logger', None)
    param_copy.pop('verbose', None)

    return AgentFactory.create_agent(agent_type=agent_type, env=env, agent_params=params)


def create_rl_agent_catch(params: Dict[str, Any]) -> Agent:
    """
    Create RL agent for the Catch environment.
    """
    param_copy = dict(params)
    agent_type = param_copy.pop('agent_type')
    num_episodes = param_copy.pop('num_episodes')
    # the remaining parameters are the agent parameters.
    return create_agent_for_catch_env(agent_type, num_episodes, agent_params=params)
