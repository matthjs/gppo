from typing import Union

import gymnasium as gym
import torch

from src.agents.agent import Agent
from src.agents.ddqnagent import DDQNAgent
from src.agents.dqnagent import DQNAgent
from src.agents.dqvagent import DQVAgent
from src.agents.dqvmaxagent import DQVMaxAgent
from src.agents.gppoagent import GPPOAgent
from src.agents.gpreinforceagent import GPReinforceAgent
from src.agents.randomagent import RandomAgent
from src.agents.reinforceagent import ReinforceAgent
from src.agents.ppoagent import PPOAgent


class AgentFactory:
    # TODO: Find a better way to do this
    DGP_ARCHITECTURES = {
        0: {
            "hidden_layers_config": [{"output_dims": 1, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
        },
        1: {
            "hidden_layers_config": [{"output_dims": 64, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": 64, "mean_type": "linear"}],
            "value_hidden_config": [{"output_dims": 64, "mean_type": "linear"}],
        },
        2: {
            "hidden_layers_config": [{"output_dims": 128, "mean_type": "constant"}],
            "policy_hidden_config": [{"output_dims": 128, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": 128, "mean_type": "constant"}],
        },
    }

    @staticmethod
    def create_agent(agent_type: str,
                     env: Union[gym.Env, str],
                     agent_params: dict,
                     device=None) -> Agent:
        """
        Create an agent of agent_type with agent_params.
        """
        if isinstance(env, str):
            env = gym.make(env)  # Env params?

        obs_space = env.observation_space
        action_space = env.action_space
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "")

        agent_params.update({
            "state_dimensions": obs_space.shape,
            "action_dimensions": action_space.shape,
            "device": device,
        })

        if agent_type == "DQN":
            # This looks like syntactic nonsense but basically we want to overwrite the class
            # type and name so that the dueling architecture variant is treated as a separate agent_type.
            return DQNAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DuelingDQNAgent", (DQNAgent,), {})(**agent_params)
        elif agent_type == "DDQN":
            return DDQNAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DuelingDDQNAgent", (DDQNAgent,), {})(**agent_params)
        elif agent_type == "DQV":
            return DQVAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DuelingDQVAgent", (DQVAgent,), {})(**agent_params)
        elif agent_type == "DQV-Max":
            return DQVMaxAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DQVMaxAgent", (DQVMaxAgent,), {})(**agent_params)
        elif agent_type == "GPReinforce":
            return GPReinforceAgent(
                **agent_params)  # I don't think this runs anymore due to outdated interface somehwere.
        elif agent_type == "Reinforce":
            return ReinforceAgent(**agent_params)
        elif agent_type == "PPO":
            return PPOAgent(**agent_params)
        elif agent_type == "GPPO":
            # Check if architecture_choice is present
            arch_choice = agent_params.pop("architecture_choice", None)
            if arch_choice is not None:
                arch_config = AgentFactory.DGP_ARCHITECTURES.get(arch_choice)
                if arch_config is None:
                    raise ValueError(f"Invalid architecture_choice {arch_choice} for GPPOAgent.")
                # Update agent_params with architecture config
                agent_params.update(arch_config)
            return GPPOAgent(**agent_params)
        elif agent_type == "RANDOM":
            return RandomAgent(action_space)

        raise ValueError("Invalid agent type")
