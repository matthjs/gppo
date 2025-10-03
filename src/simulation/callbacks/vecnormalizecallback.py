import os
import logging
from typing import Optional, Union, cast
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from stable_baselines3.common.vec_env import VecNormalize


class VecNormalizeCallback(AbstractCallback):
    """
    Save/load VecNormalize stats using SB3 API exclusively.

    Directory layout:
      <save_base>/<exp_id>/<env_id>/<agent_id>/<run_id>/obs_norm_stats.pkl
    Top-level:
      <save_base>/<agent_id><env_id>run_<run_id>obs_norm_stats.pkl
    """

    def __init__(
        self,
        save_base: str = "./results",
        run_id: Union[str, int] = 0,
        load_on_start: bool = True,
        save_on_end: bool = True,
        also_save_top_level: bool = False,
    ) -> None:
        super().__init__()
        self.save_base = save_base
        self.run_id = str(run_id)
        self.load_on_start = load_on_start
        self.save_on_end = save_on_end
        self.also_save_top_level = also_save_top_level

        # populated in init_callback
        self.agent_id: Optional[str] = None
        self.env_id: Optional[str] = None
        self.env_manager = None
        self.exp_id: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.run_filepath: Optional[str] = None
        self.top_filepath: Optional[str] = None

    def init_callback(self, data: SimulatorRLData) -> None:
        super().init_callback(data)
        # keep the existing behavior for agent_id fallback
        self.agent_id = getattr(data, "agent_id", getattr(data, "agent", None) and type(data.agent).__name__ or "agent")
        self.env_id = getattr(data, "env_id", "env")
        self.env_manager = getattr(data, "env_manager", None)
        self.exp_id = getattr(data, "experiment_id")
        # prefer save_base from data if present
        self.save_base = getattr(data, "save_path", self.save_base)

        # run_dir: results/<exp_id>/<env_id>/<agent_id>/<run_id>/
        self.run_dir = os.path.join(self.save_base, self.exp_id, self.env_id, self.agent_id, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.run_filepath = os.path.join(self.run_dir, "obs_norm_stats.pkl")
        self.top_filepath = os.path.join(self.save_base, f"{self.agent_id}{self.env_id}run_{self.run_id}obs_norm_stats.pkl")

    def _get_vecnormalize(self) -> VecNormalize:
        """
        Return the VecNormalize instance directly. Assumes env_manager and vec_env exist
        and that vec_env is a VecNormalize instance. Let exceptions propagate if not.
        """
        # direct access, allow AttributeError if env_manager or vec_env missing
        return cast(VecNormalize, self.env_manager.vec_env)

    def _save_vecnorm_to(self, vecnorm: VecNormalize, path: str) -> None:
        """
        Save using SB3 VecNormalize.save(). Assumes vecnorm.save exists and works.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        vecnorm.save(path)
        logger.info("Saved VecNormalize via save() -> %s", path)

    def _load_vecnorm_from(self, vecnorm: VecNormalize, path: str) -> bool:
        """
        Load using SB3 VecNormalize.load(load_path, venv). Assumes vecnorm has .venv.
        Returns True on success, False if file not present. Raises on other failures.
        """
        if not os.path.exists(path):
            return False

        # assume vecnorm is a VecNormalize and has .venv attribute
        base_venv = vecnorm.venv
        loaded = VecNormalize.load(path, base_venv)

        # replace manager's vec_env with the loaded wrapper and preserve training flag
        self.env_manager.vec_env = loaded
        loaded.training = bool(self.env_manager.training)

        logger.info("Loaded VecNormalize via VecNormalize.load() from %s", path)
        return True

    def on_training_start(self) -> None:
        super().on_training_start()
        if not self.load_on_start:
            return

        vecnorm = self._get_vecnormalize()
        # try run-specific file first, then top-level file
        if self._load_vecnorm_from(vecnorm, self.run_filepath):
            return
        if self._load_vecnorm_from(vecnorm, self.top_filepath):
            return

        logger.info("No normalization file found for run (%s) or top-level, skipping load.", self.run_filepath)

    def on_training_end(self) -> None:
        super().on_training_end()
        if not self.save_on_end:
            return

        vecnorm = self._get_vecnormalize()

        # ensure directories exist
        os.makedirs(os.path.dirname(self.top_filepath), exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

        # save both places: run folder and top-level (if requested)
        self._save_vecnorm_to(vecnorm, self.run_filepath)
        if self.also_save_top_level:
            self._save_vecnorm_to(vecnorm, self.top_filepath)
