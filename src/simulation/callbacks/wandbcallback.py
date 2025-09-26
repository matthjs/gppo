import logging
import pathlib
import wandb
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData
from src.simulation.callbacks.metrictrackercallback import MetricTrackerCallback

logger = logging.getLogger(__name__)


class WandbCallback(AbstractCallback):
    """
    Wraps a MetricTrackerCallback and logs its metrics to Weights & Biases (wandb).

    - Delegates all metric tracking to MetricTrackerCallback.
    - Logs episode-level metrics by default.
    - Step-level logging optional if the wrapped callback exposes step metrics.
    """

    def __init__(
        self,
        metric_callback: MetricTrackerCallback,
        project: str,
        entity: str = None,
        config: dict = None,
        run_name: str = None,
        verbose: int = logging.INFO,
        plot_dir: str = None
    ):
        super().__init__()
        self.metric_callback = metric_callback
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.config = config or {}
        self.plot_dir = pathlib.Path(plot_dir) if plot_dir else None
        self.run = None

        logger.setLevel(verbose)

    def init_callback(self, data: SimulatorRLData):
        """
        Initialize both the MetricTrackerCallback and a wandb run.
        """
        super().init_callback(data)
        self.metric_callback.init_callback(data)

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=self.config,
            reinit=True,  # allow multiple runs in the same process
        )

        logger.info(f"WandbCallback initialized (project={self.project}, run={self.run_name})")

    def on_step(self, action, reward, next_obs, done) -> bool:
        """
        Forward step to MetricTrackerCallback and optionally log step metrics.
        """
        super().on_step(action, reward, next_obs, done)

        # Delegate metric tracking
        continue_training = self.metric_callback.on_step(action, reward, next_obs, done)

        # Optional step-level logging
        # Example: log running returns per environment if you want
        # if hasattr(self.metric_callback, "episode_returns"):
        #     for i in range(self.metric_callback.n_env):
        #         wandb.log({f"env_{i}_running_return": self.metric_callback.episode_returns[i]})

        return continue_training
    
    def _log_plots(self):
        """
        Log all .png/.svg/.pdf files from the plot directory to wandb.
        """
        if not self.plot_dir or not self.plot_dir.exists():
            return

        for ext in ("*.png", "*.svg", "*.pdf"):
            for file in self.plot_dir.glob(ext):
                wandb.log({f"plots/{file.name}": wandb.Image(str(file))})
                logger.debug(f"Logged plot {file}")

    def on_episode_end(self):
        super().on_episode_end()
        self.metric_callback.on_episode_end()

        # Log episode-level metrics
        if len(self.metric_callback.completed_returns) > 0:
            last_return = self.metric_callback.completed_returns[-1]
            wandb.log({
                "episode": self.metric_callback.episodes_finished,
                "return": last_return,
            })
            logger.debug(
                f"WandbCallback logged episode {self.metric_callback.episodes_finished}, return={last_return:.2f}"
            )
        
        # Log plots at the end of each episode
        self._log_plots()

    def on_learn(self, learning_info):
        """
        Log learning info metrics from the agent to wandb.
        """
        # Forward to MetricTrackerCallback
        self.metric_callback.on_learn(learning_info)

        if learning_info:
            wandb.log(learning_info)

    def on_training_end(self):
        super().on_training_end()
        self.metric_callback.on_training_end()

        # Final log of plots
        self._log_plots()

        wandb.finish()
        logger.info("WandbCallback training ended and run closed.")
