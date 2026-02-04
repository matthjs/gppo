import os
import logging
from gppo.simulation.callbacks.abstractcallback import AbstractCallback
from gppo.agents.agent import Agent

logger = logging.getLogger(__name__)


class AgentSaveCallback(AbstractCallback):
    """
    Callback to save the RL agent at the end of training.

    Parameters
    ----------
    agent: Agent
        The RL agent to save.
    save_path: str
        Directory to save the agent.
    run_id: int
        Identifier for the run (used in filenames).
    save_model: bool
        Whether to save the agent model.
    verbose: int
        Logging level.

    Example
    -------
        >>> save_cb = AgentSaveCallback(agent, save_path="./saves", run_id=1)
        >>> save_cb.on_training_end()
    """

    def __init__(self, agent: Agent, save_path: str = "./", run_id: int = 0, save_model: bool = True, verbose: int = logging.INFO):
        super().__init__()
        self.agent = agent
        self.save_path = save_path
        self.run_id = run_id
        self.save_model = save_model
        logger.setLevel(verbose)

    def on_training_end(self, env_id: str = ""):
        """
        Save the agent model.
        """
        if self.save_model:
            model_path = os.path.join(self.save_path, f"{type(self.agent).__name__}{env_id}_run_{self.run_id}.pt")
            logger.info(f"Saving agent to {model_path} ...")
            self.agent.save(model_path)
