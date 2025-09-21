import os
import logging
from src.simulation.callbacks.abstractcallback import AbstractCallback

logger = logging.getLogger(__name__)


class EnvStatsSaveCallback(AbstractCallback):
    """
    Callback to save environment normalization statistics (VecNormalize) at the end of training.

    Parameters
    ----------
    save_path: str
        Directory to save the stats.
    run_id: int
        Identifier for the run (used in filenames).
    verbose: int
        Logging level.

    Example
    -------
        >>> env_stats_cb = EnvStatsSaveCallback(save_path="./saves", run_id=1)
        >>> env_stats_cb.on_training_end(env)
    """

    def __init__(self, save_path: str = "./", run_id: int = 0, verbose: int = logging.INFO):
        super().__init__()
        self.save_path = save_path
        self.run_id = run_id
        logger.setLevel(verbose)

    def on_training_end(self, env):
        """
        Save VecNormalize statistics of the environment.
        """
        if env and hasattr(env, "save"):
            stats_path = os.path.join(self.save_path, f"{type(env).__name__}_run_{self.run_id}_obs_norm_stats.pkl")
            logger.info(f"Saving environment normalization stats to {stats_path} ...")
            env.save(stats_path)
