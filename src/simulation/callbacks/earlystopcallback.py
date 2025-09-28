import logging
from collections import deque
import numpy as np
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)

class EarlyStopCallback(AbstractCallback):
    """
    Early stopping callback based on average return over a window.
    NOTE: The return tracking is now repeated at three different places which is not ideal.
    """

    def __init__(self, early_stop_check: int, early_stop_window: int, early_stop_threshold: float, verbose=logging.INFO):
        super().__init__()
        self.early_stop_check = early_stop_check
        self.early_stop_window = early_stop_window
        self.early_stop_threshold = early_stop_threshold
        self.reward_buffer = deque(maxlen=early_stop_window)
        self.stop_triggered = False
        self.episodes_finished = 0
        self.episode_returns = None
        self.n_env = None
        logging.getLogger(__name__).setLevel(verbose)

    def init_callback(self, data: SimulatorRLData) -> None:
        super().init_callback(data)
        self.n_env = data.n_env
        self.episode_returns = np.zeros(self.n_env, dtype=np.float32)
        self.episodes_finished = 0

    def on_step(self, action, reward, next_obs, done) -> bool:
        super().on_step(action, reward, next_obs, done)
        self.episode_returns += reward

        for i in range(self.n_env):
            if done[i]:
                self.reward_buffer.append(self.episode_returns[i])
                self.episode_returns[i] = 0.0
                self.episodes_finished += 1

                if (
                    self.episodes_finished % self.early_stop_check == 0
                    and len(self.reward_buffer) == self.reward_buffer.maxlen
                ):
                    avg_recent = np.mean(self.reward_buffer)
                    logger.info(
                        f"Early stop check at episode {self.episodes_finished}: "
                        f"avg return={avg_recent:.2f}, threshold={self.early_stop_threshold}"
                    )
                    if avg_recent < self.early_stop_threshold:
                        logger.warning(f"Early stopping triggered at episode {self.episodes_finished}")
                        self.stop_triggered = True
                        return False

        return not self.stop_triggered

    def on_episode_end(self) -> None:
        super().on_episode_end()

    def on_training_end(self) -> None:
        super().on_training_end()
