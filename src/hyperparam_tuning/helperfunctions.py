from typing import Optional, Union
from src.agents.agentfactory import AgentFactory
from src.simulation.simulator_rl import SimulatorRL
from typing import Dict, Any
from src.agents.agent import Agent
import gymnasium as gym


from typing import Any, Dict, Union
import gymnasium as gym
from src.agents.agent import Agent
from src.agents.agentfactory import AgentFactory
from src.simulation.envmanager import EnvManager
from src.simulation.callbacks.abstractcallback import AbstractCallback


def train_rl_agent(agent: Agent, params: Dict[str, Any], env_id: str,
                   exp_id: str = "hyperopt_exp",
                   agent_id: Optional[str] = None,
                   callbacks: Optional[list[AbstractCallback]] = None,
                   normalize_obs: bool = True) -> Dict[str, float]:
    """
    Train the RL agent using the SimulatorRL wrapper.

    :param agent: RL Agent to train.
    :param params: Training hyperparameters.
    :param env: The environment to train in (Gym API).
    :param callbacks: Optional list of training callbacks.
    :param normalize_obs: Whether to normalize observations.
    :return: Dictionary with training metrics.
    """
    env_manager = EnvManager(env_id=env_id,
                             env_fn=lambda: gym.make(env_id),
                             n_envs=params.get("n_envs", 1),
                             norm_obs=normalize_obs)
    simulator = SimulatorRL(
        experiment_id=exp_id,
        agent_id=agent_id if agent_id is not None else type(agent).__name__,
        env_manager=env_manager,
        agent=agent,
        num_episodes=params["num_episodes"],
        callbacks=callbacks,
        save_model=False,
        load_model=False,
        device=params.get("device", None),
    )

    avg_return = simulator.train()

    return {"train/avg_return": avg_return} if avg_return is not None else {}


def eval_rl_agent(agent: Agent, params: Dict[str, Any], env_id: str,
                  exp_id: str = "hyperopt_exp",
                  agent_id: Optional[str] = None,
                  callbacks: Optional[list[AbstractCallback]] = None,
                  normalize_obs: bool = True) -> Dict[str, float]:
    """
    Evaluate the RL agent (no learning updates).

    :param agent: RL Agent to evaluate.
    :param params: Evaluation hyperparameters.
    :param env: The environment to evaluate in (Gym API).
    :param callbacks: Optional list of callbacks.
    :param normalize_obs: Whether to normalize observations.
    :return: Dictionary with evaluation metrics.
    """
    env_manager = EnvManager(env_id=env_id,
                             env_fn=lambda: gym.make(env_id),
                             n_envs=params.get("n_envs", 1),
                             norm_obs=normalize_obs)

    simulator = SimulatorRL(
        experiment_id=exp_id,
        agent_id=agent_id if agent_id is not None else type(agent).__name__,
        env_manager=env_manager,
        agent=agent,
        num_episodes=-1,  # No training episodes
        callbacks=callbacks,
        save_model=False,
        load_model=False,
        device=params.get("device", None),
    )

    avg_return = simulator.evaluate(num_eval_episodes=params.get("num_eval_episodes", 10))
    return {"return": avg_return}


def create_rl_agent(params: Dict[str, Any], env: Union[str, gym.Env]) -> Agent:
    """
    Create an RL agent using the AgentFactory with a specified environment.

    :param params: Dictionary of agent parameters (must include 'agent_type').
    :param env: A Gym environment instance or string name.
    :return: Instantiated Agent.
    """
    param_copy = dict(params)
    agent_type = param_copy.pop("agent_type")
    n_envs = param_copy.pop("n_envs")

    # Remove non-agent params
    for key in ["num_episodes", "num_eval_episodes", "wandb_logger",
                "verbose", "save_model", "load_model",
                "experiment_id", "agent_id", "device"]:
        param_copy.pop(key, None)

    return AgentFactory.create_agent(agent_type=agent_type,
                                     env=env,
                                     n_envs=n_envs,
                                     agent_params=param_copy)
