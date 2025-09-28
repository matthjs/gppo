import os
import pickle
import logging
from typing import Optional, Union
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AgentCheckpointCallback(AbstractCallback):
    """
    Save / load agent checkpoints following the directory conventions:
      <save_base>/<env_id>/<agent_id>/<run_id>/<agent_id>_run_<run_id>.pt
    and also (optionally) a top-level fallback:
      <save_base>/<agent_id><env_id>run_<run_id>.pt
    """

    def __init__(
        self,
        save_base: str = "./results",
        run_id: Union[str, int] = 0,
        load_on_start: bool = True,
        save_on_end: bool = True,
        save_freq_episodes: Optional[int] = None,
        also_save_top_level: bool = False,
    ) -> None:
        super().__init__()
        self.save_base = save_base
        self.run_id = str(run_id)
        self.load_on_start = load_on_start
        self.save_on_end = save_on_end
        self.save_freq_episodes = save_freq_episodes
        self.also_save_top_level = also_save_top_level

        # populated in init_callback
        self.agent = None
        self.agent_id = None
        self.env_id = None
        self.run_dir = None
        self.run_filepath = None
        self.top_filepath = None
        self.episodes_finished = 0

    def init_callback(self, data: SimulatorRLData) -> None:
        super().init_callback(data)
        self.agent = getattr(data, "agent", None)
        self.agent_id = getattr(data, "agent_id", self.agent and type(self.agent).__name__ or "agent")
        self.env_id = getattr(data, "env_id")
        self.exp_id = getattr(data, "experiment_id")
        self.save_base = getattr(data, "save_path", self.save_base)

        # create run directory and filepaths
        self.run_dir = os.path.join(self.save_base, self.exp_id, self.env_id, self.agent_id, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_filepath = os.path.join(self.run_dir, f"{self.agent_id}_run_{self.run_id}.pt")
        self.top_filepath = os.path.join(self.save_base, f"{self.agent_id}{self.env_id}run_{self.run_id}.pt")

    def _save_agent_to(self, path: str) -> None:
        if self.agent is None:
            logger.warning("AgentCheckpointCallback: no agent to save.")
            return
        try:
            # expect agent.save(filepath) to exist
            self.agent.save(path)
            logger.info("Saved agent to %s", path)
        except Exception as e:
            logger.exception("Failed to save agent to %s: %s", path, e)

    def _load_agent_from(self, path: str) -> bool:
        if self.agent is None:
            logger.warning("AgentCheckpointCallback: no agent to load into.")
            return False
        if not os.path.exists(path):
            return False
        try:
            self.agent.load(path)
            logger.info("Loaded agent from %s", path)
            return True
        except Exception as e:
            logger.exception("Failed to load agent from %s: %s", path, e)
            return False

    def on_training_start(self) -> None:
        super().on_training_start()
        if not self.load_on_start:
            return

        # try run-specific first, then top-level fallback
        if self._load_agent_from(self.run_filepath):
            return
        if self.also_save_top_level and self._load_agent_from(self.top_filepath):
            return

        logger.info("No agent checkpoint found for run (%s) or top-level, skipping load.", self.run_filepath)

    def on_episode_end(self) -> None:
        super().on_episode_end()
        # increment local episodes counter
        self.episodes_finished += 1

        if self.save_freq_episodes and self.episodes_finished > 0 and (self.episodes_finished % self.save_freq_episodes == 0):
            # ensure dir exists
            os.makedirs(os.path.dirname(self.run_filepath), exist_ok=True)
            self._save_agent_to(self.run_filepath)
            if self.also_save_top_level:
                self._save_agent_to(self.top_filepath)

    def on_training_end(self) -> None:
        super().on_training_end()
        if self.save_on_end:
            os.makedirs(os.path.dirname(self.run_filepath), exist_ok=True)
            self._save_agent_to(self.run_filepath)
            if self.also_save_top_level:
                self._save_agent_to(self.top_filepath)
            logger.info("Saved agent at end of training.")
