from abc import abstractmethod, ABC
import torch
import numpy as np
from typing import Tuple, Callable, Any, Dict, Optional
from src.agents.agent import Agent
from src.util.exploration import ExplorationPolicy
from src.util.replaybuffer import ReplayBuffer


class OffPolicyAgent(Agent, ABC):
    def __init__(
            self,
            memory_size: int,
            state_dimensions: Tuple[int, int, int],
            n_actions: int,
            batch_size: int,
            learning_rate: float,
            discount_factor: float,
            expl_policy_factory: Callable[[], ExplorationPolicy],
            device: torch.device
            # Add any other arguments you need here
            # e.g. learning rate, discount factor, etc.
    ) -> None:
        """!
        Initializes the agent.
        Agent is an abstract class that should be inherited by any agent that
        wants to interact with the environment. The agent should be able to
        store transitions, choose actions based on observations, and learn from the
        transitions.

        @param memory_size (int): Size of the memory buffer
        @param state_dimensions (int): Number of dimensions of the state space
        @param n_actions (int): Number of actions the agent can take
        """

        self.memory_size = memory_size
        self.replay_buffer = ReplayBuffer(capacity=self.memory_size, device=device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate,
        self.discount_factor = discount_factor
        self.device = device
        """
        In child class:
            policy_factory = make_policy(
                name="epsilon_greedy",
                epsilon=0.1,
            )    
            super().__init__(
                **kwargs,
                expl_policy_factory=policy_factory
            )
        """
        self.expl_policy = expl_policy_factory()

    def store_transition(
            self,
            state: np.ndarray,
            action: int,  # Is this always an int?
            reward: float,
            new_state: np.ndarray,
            done: bool
    ) -> None:
        """!
        Stores the state transition for later memory replay.
        Make sure that the memory buffer does not exceed its maximum size.

        Hint: after reaching the limit of the memory buffer, maybe you should start overwriting
        the oldest transitions?

        @param state        (list): Vector describing current state
        @param action       (int): Action taken
        @param reward       (float): Received reward
        @param new_state    (list): Newly observed state.
        """

        self.replay_buffer.push(
            (state, action, reward, new_state, done)
        )

    def choose_action(
            self,
            observation: np.ndarray
    ) -> int:  # Is this always an int?
        """!
        Abstract method that should be implemented by the child class, e.g. DQN or DDQN agents.
        This method should contain the full logic needed to choose an action based on the current state.
        Maybe you can store the neural network in the agent class and use it here to decide which action to take?

        @param observation (np.ndarray): Vector describing current state

        @return (int): Action to take
        """
        return self.expl_policy.select_action(observation)

    def update(self, params: Optional[Dict[str, Any]] = None):
        """
        An abstract update method that can be used to update some parts
        of the agent that may not have to do with updating model parameters.
        I.e., scheduling for epsilon greedy.
        """
        pass

    @abstractmethod
    def learn(self) -> Dict[str, Any]:
        """!
        Update the parameters of the internal networks.
        This method should be implemented by the child class.
        Can return a dictionary e.g. {'loss': 0.1}.
        """

        pass
