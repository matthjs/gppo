import gymnasium as gym
from src.agents.agent import Agent
from src.agents.ddqnagent import DDQNAgent
from src.agents.dqnagent import DQNAgent
from src.agents.dqvagent import DQVAgent
from src.agents.dqvmaxagent import DQVMaxAgent
from src.agents.gpreinforceagent import GPReinforceAgent
from src.agents.randomagent import RandomAgent


class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str,
                     env: gym.Env,
                     agent_params: dict) -> Agent:
        """
        Create an agent of agent_type with agent_params.
        """
        obs_space = env.observation_space
        action_space = env.action_space

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
            return GPReinforceAgent(**agent_params)
        elif agent_type == "RANDOM":
            return RandomAgent(action_space)

        raise ValueError("Invalid agent type")
