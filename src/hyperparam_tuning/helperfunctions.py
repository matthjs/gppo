from typing import Dict, Any
from src.agents.agent import Agent
from src.util.interaction import agent_env_loop, create_agent_for_catch_env


def train_rl_agent(agent: Agent, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Train the RL agent with the given hyperparameters.

    :param agent: RL Agent to train.
    :param params: The hyperparameters to use during training.
    :return: The average return over the number of episodes.
    """
    # defaults to Catch environment
    agent_env_loop(agent,
                   params['num_episodes'],
                   learning=True)
    # Could maybe return something here.
    return {}


def eval_rl_agent(agent: Agent, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate the RL agent, e.g., let it interact with the environment without
    performing any learning update.

    :param agent: RL Agent to evaluate.
    :param params: Agent parameters (e.g., memory_size, learning rate etc.)
    """
    ret = agent_env_loop(agent,
                         params['num_eval_episodes'],
                         learning=False)
    return {"return": ret}


def create_rl_agent_catch(params: Dict[str, Any]) -> Agent:
    """
    Create RL agent for the Catch environment.
    """
    param_copy = dict(params)
    agent_type = param_copy.pop('agent_type')
    num_episodes = param_copy.pop('num_episodes')
    # the remaining parameters are the agent parameters.
    return create_agent_for_catch_env(agent_type, num_episodes, agent_params=params)