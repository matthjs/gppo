from typing import Union, Optional
import gymnasium as gym
import torch
from src.agents.agent import Agent
from src.agents.ddqnagent import DDQNAgent
from src.agents.dqnagent import DQNAgent
from src.agents.dqvagent import DQVAgent
from src.agents.dqvmaxagent import DQVMaxAgent
from src.agents.gppoagent import GPPOAgent
from src.agents.randomagent import RandomAgent
from src.agents.ppoagent import PPOAgent


class AgentFactory:
    """
    Simple factory for creating an agent of agent_type from agent_params.
    """
    # Static member variable. Used in hyperparameter tuning to map integers to model architectures.
    DGP_ARCHITECTURES = {
        0: {
            "hidden_layers_config": [{"output_dims": 1, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
        },
        1: {
            "hidden_layers_config": [{"output_dims": 1, "mean_type": "linear"},
                                     {"output_dims": 1, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
        },
        2: {
            "hidden_layers_config": [{"output_dims": 1, "mean_type": "linear"},
                                     {"output_dims": 1, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": 1, "mean_type": "linear"},
                                     {"output_dims": None, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": 1, "mean_type": "linear"},
                                    {"output_dims": None, "mean_type": "constant"}],
        },
        3: {
            "hidden_layers_config": [{"output_dims": 6, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": 6, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
        },
        4: {
            "hidden_layers_config": [{"output_dims": 1, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": 6, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": None, "mean_type": "constant"}],
        },
        5: {
            "hidden_layers_config": [{"output_dims": 6, "mean_type": "linear"},
                                     {"output_dims": 6, "mean_type": "linear"}],
            "policy_hidden_config": [{"output_dims": 6, "mean_type": "linear"},
                                     {"output_dims": 6, "mean_type": "constant"}],
            "value_hidden_config": [{"output_dims": 1, "mean_type": "linear"},
                                    {"output_dims": None, "mean_type": "constant"}],
        }
    }
    def __init__(self):
        pass

    @staticmethod
    def create_agent(agent_type: str,
                     env: Union[gym.Env, str],
                     agent_params: dict,
                     load_model: bool = False,
                     model_path: Optional[str] = None,
                     device=None) -> Agent:
        """
        Create an agent of agent_type with agent_params.
        :param agent_type: What agent to instantiate (e.g., PPO, GPPO).
        :param env: Environment to train the RL agent on. This is used to pass on information such
        as the size of the state space, number of actions.
        :param agent_params: Arguments to pass to the RL algorithm constructors.
        :param load_model: Whether to load the model or not.
        :param model_path: Pass a path to load model parameters from.
        :param device: E.g., GPU (cuda:0) or CPU (cpu).
        """
        if isinstance(env, str):
            env = gym.make(env)

        obs_space = env.observation_space
        action_space = env.action_space
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "")

        agent_params.update({
            "state_dimensions": obs_space.shape,
            "action_dimensions": action_space.shape,
            "device": device,
        })

        agent = None

        if agent_type == "DQN":
            # This looks like syntactic nonsense, but basically we want to overwrite the class
            # type and name so that the dueling architecture variant is treated as a separate agent_type.
            agent = DQNAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DuelingDQNAgent", (DQNAgent,), {})(**agent_params)
        elif agent_type == "DDQN":
            agent = DDQNAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DuelingDDQNAgent", (DDQNAgent,), {})(**agent_params)
        elif agent_type == "DQV":
            agent = DQVAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DuelingDQVAgent", (DQVAgent,), {})(**agent_params)
        elif agent_type == "DQV-Max":
            agent = DQVMaxAgent(**agent_params) if not agent_params['dueling_architecture'] else \
                type("DQVMaxAgent", (DQVMaxAgent,), {})(**agent_params)
        elif agent_type == "PPO":
            agent = PPOAgent(**agent_params)
        elif agent_type == "GPPO":
            # Check if architecture_choice is present
            arch_choice = agent_params.pop("architecture_choice", None)
            if arch_choice is not None:
                arch_config = AgentFactory.DGP_ARCHITECTURES.get(arch_choice)
                if arch_config is None:
                    raise ValueError(f"Invalid architecture_choice {arch_choice} for GPPOAgent.")
                # Update agent_params with architecture config
                agent_params.update(arch_config)
            agent = GPPOAgent(**agent_params)
        elif agent_type == "RANDOM":
            agent = RandomAgent(action_space)
        else:
            raise ValueError("Invalid agent type")

        if load_model:
            # loads model weights
            agent.load(model_path)

        return agent

