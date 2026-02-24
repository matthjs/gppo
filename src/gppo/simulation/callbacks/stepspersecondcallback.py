import time
import logging
import numpy as np
from collections import deque
from typing import Dict, Any

from gppo.simulation.callbacks.abstractcallback import AbstractCallback
from gppo.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)


class StepsPerSecondCallback(AbstractCallback):
    """
    Measures and logs steps per second during training.

    Tracks both instantaneous SPS (over a rolling window) and
    overall SPS (since training start), so you can see both
    peak throughput and long-term average.
    """

    def __init__(
        self,
        window_size: int = 1000,
        log_frequency: int = 500,
        verbose: int = logging.INFO,
    ):
        """
        Args:
            window_size:    Number of recent timesteps to use for the rolling SPS estimate.
            log_frequency:  How often (in total timesteps) to print SPS to the logger.
            verbose:        Logging level.
        """
        super().__init__()
        self.window_size = window_size
        self.log_frequency = log_frequency

        # Ring buffer of (timestamp, n_steps_at_that_point) tuples
        self._window: deque = deque()

        self._training_start_time: float = 0.0
        self._total_timesteps: int = 0
        self._n_env: int = 1

        # Exposed metrics (available after training or at any point during)
        self.current_sps: float = 0.0
        self.overall_sps: float = 0.0
        self.peak_sps: float = 0.0

        logger.setLevel(verbose)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_callback(self, data: SimulatorRLData) -> None:
        super().init_callback(data)
        self._n_env = data.n_env
        self._training_start_time = time.perf_counter()
        self._window.clear()
        self._window.append((self._training_start_time, 0))
        logger.info(f"StepsPerSecondCallback initialised with {self._n_env} env(s).")

    def on_step(self, action, reward, next_obs, done) -> bool:
        super().on_step(action, reward, next_obs, done)

        self._total_timesteps += self._n_env
        now = time.perf_counter()

        # Maintain the rolling window
        self._window.append((now, self._total_timesteps))
        while len(self._window) > self.window_size:
            self._window.popleft()

        # Rolling SPS
        oldest_time, oldest_steps = self._window[0]
        elapsed_window = now - oldest_time
        if elapsed_window > 0:
            self.current_sps = (self._total_timesteps - oldest_steps) / elapsed_window
        
        # Overall SPS
        total_elapsed = now - self._training_start_time
        if total_elapsed > 0:
            self.overall_sps = self._total_timesteps / total_elapsed

        if self.current_sps > self.peak_sps:
            self.peak_sps = self.current_sps

        if self._total_timesteps % self.log_frequency == 0:
            logger.info(
                f"[Step {self._total_timesteps:>10,}] "
                f"SPS (rolling): {self.current_sps:>8,.0f} | "
                f"SPS (overall): {self.overall_sps:>8,.0f} | "
                f"Peak: {self.peak_sps:>8,.0f}"
            )

        return True

    def on_training_end(self) -> None:
        super().on_training_end()
        total_elapsed = time.perf_counter() - self._training_start_time
        logger.info(
            f"\n{'='*55}\n"
            f"  Training finished\n"
            f"  Total timesteps : {self._total_timesteps:,}\n"
            f"  Total time      : {total_elapsed:.1f}s\n"
            f"  Overall SPS     : {self.overall_sps:,.0f}\n"
            f"  Peak SPS        : {self.peak_sps:,.0f}\n"
            f"{'='*55}"
        )
