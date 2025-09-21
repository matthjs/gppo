import logging
from collections import deque
from src.simulation.callbacks.abstractcallback import AbstractCallback

logger = logging.getLogger(__name__)

class EarlyStopCallback(AbstractCallback):
    """
    Early stopping callback based on average return over a window.
    """

    def __init__(self, early_stop_check: int, early_stop_window: int, early_stop_threshold: float, verbose=logging.INFO):
        super().__init__()
        self.early_stop_check = early_stop_check
        self.early_stop_window = early_stop_window
        self.early_stop_threshold = early_stop_threshold
        self.reward_buffer = deque(maxlen=early_stop_window)
        self.stop_triggered = False
        self.episodes_finished = 0
        logging.getLogger(__name__).setLevel(verbose)

    def on_episode_end(self, episode_return: float) -> bool:
        """
        Should be called at the end of each episode.
        Returns False if early stopping is triggered.
        """
        self.episodes_finished += 1
        self.reward_buffer.append(episode_return)

        if self.episodes_finished % self.early_stop_check == 0 and len(self.reward_buffer) == self.reward_buffer.maxlen:
            avg_recent = sum(self.reward_buffer) / len(self.reward_buffer)
            logger.info(f"Early stop check at episode {self.episodes_finished}: avg return={avg_recent:.2f}, threshold={self.early_stop_threshold}")
            if avg_recent < self.early_stop_threshold:
                logger.warning(f"Early stopping triggered at episode {self.episodes_finished}")
                self.stop_triggered = True
                return False

        return True
