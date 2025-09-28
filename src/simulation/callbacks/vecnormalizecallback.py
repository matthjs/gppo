import os
import pickle
import logging
from typing import Optional, Union
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from stable_baselines3.common.vec_env import VecNormalize


class VecNormalizeCallback(AbstractCallback):
    """
    Save/load VecNormalize stats following the directory conventions:
      <save_base>/<env_id>/<agent_id>/<run_id>/obs_norm_stats.pkl
    and also save a top-level file:
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

        # filled in by init_callback
        self.agent_id = None
        self.env_id = None
        self.env_manager = None

    def init_callback(self, data: SimulatorRLData) -> None:
        super().init_callback(data)
        self.agent_id = getattr(data, "agent_id", getattr(data, "agent", None) and type(data.agent).__name__ or "agent")
        self.env_id = getattr(data, "env_id", "env")
        self.env_manager = getattr(data, "env_manager", None)
        self.exp_id = getattr(data, "experiment_id")
        # prefer save_base from data if present (keeps tests consistent)
        self.save_base = getattr(data, "save_path", self.save_base)
        # run_dir: results/<env_id>/<agent_id>/<run_id>/
        self.run_dir = os.path.join(self.save_base, self.exp_id, self.env_id, self.agent_id, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        # run file path and top-level file path
        self.run_filepath = os.path.join(self.run_dir, "obs_norm_stats.pkl")
        self.top_filepath = os.path.join(self.save_base, f"{self.agent_id}{self.env_id}run_{self.run_id}obs_norm_stats.pkl")

    def _get_vecnormalize(self):
        if self.env_manager is not None:
            vec_env = getattr(self.env_manager, "vec_env", None)
        else:
            vec_env = None

        if vec_env is None:
            logger.debug("VecNormalizeCallback: vec_env not available.")
            return None

        if VecNormalize is not None and isinstance(vec_env, VecNormalize):
            return vec_env

        # fallback: if has typical VecNormalize attrs assume it's compatible
        if hasattr(vec_env, "obs_rms") or hasattr(vec_env, "training"):
            return vec_env

        return None

    def _save_vecnorm_to(self, vecnorm, path: str) -> None:
        try:
            if hasattr(vecnorm, "save"):
                vecnorm.save(path)
                logger.info("Saved VecNormalize via save() -> %s", path)
                return
        except Exception:
            logger.exception("VecNormalize.save() failed; falling back to pickling obs_rms.")

        # fallback: pickle obs_rms attribute
        try:
            obs_rms = getattr(vecnorm, "obs_rms", None)
            with open(path, "wb") as f:
                pickle.dump(obs_rms, f)
            logger.info("Saved VecNormalize.obs_rms via pickle -> %s", path)
        except Exception as e:
            logger.exception("Failed to pickle obs_rms -> %s", e)

    def _load_vecnorm_from(self, vecnorm, path: str) -> bool:
        """Return True if load succeeded, False otherwise."""
        if not os.path.exists(path):
            return False
        try:
            # Preferred SB3 API: VecNormalize.load(path, venv) -> VecNormalize instance
            if VecNormalize is not None and hasattr(VecNormalize, "load"):
                try:
                    loaded = VecNormalize.load(path, vecnorm)
                    # VecNormalize.load may return a new wrapper. If we have env_manager, replace it.
                    if self.env_manager is not None:
                        self.env_manager.vec_env = loaded
                    logger.info("Loaded VecNormalize via VecNormalize.load() from %s", path)
                    return True
                except Exception:
                    # continue to fallback
                    logger.debug("VecNormalize.load() attempt failed; trying pickle fallback.")
                    pass

            # Fallback: pickle load obs_rms and assign
            with open(path, "rb") as f:
                obs_rms = pickle.load(f)
            if obs_rms is not None and hasattr(vecnorm, "obs_rms"):
                setattr(vecnorm, "obs_rms", obs_rms)
                logger.info("Loaded obs_rms via pickle and assigned to vecnorm.obs_rms from %s", path)
                return True
            else:
                logger.warning("Loaded file but couldn't assign obs_rms.")
        except Exception as e:
            logger.exception("Failed to load vecnorm from %s: %s", path, e)
        return False

    def on_training_start(self) -> None:
        super().on_training_start()
        if not self.load_on_start:
            return

        vecnorm = self._get_vecnormalize()
        if vecnorm is None:
            logger.debug("VecNormalizeCallback: no VecNormalize wrapper detected at start.")
            return

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
        if vecnorm is None:
            logger.debug("VecNormalizeCallback: no VecNormalize wrapper detected at end.")
            return

        # ensure directories exist
        os.makedirs(os.path.dirname(self.top_filepath), exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

        # save both places: run folder and top-level (matches tree)
        self._save_vecnorm_to(vecnorm, self.run_filepath)
        if self.also_save_top_level:
            # copy run file to top-level using same mechanism (or save directly)
            self._save_vecnorm_to(vecnorm, self.top_filepath)

