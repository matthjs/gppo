import wandb
import os
from typing import Optional, Dict, Any
from datetime import datetime
from gppo.metrics.metrictracker import MetricsTracker


class WandbLogger:
    """
    Wrapper for Weights & Biases (wandb) logging with optional metric tracking.
    """
    def __init__(
            self,
            enable: bool = True,
            project: Optional[str] = None,
            entity: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            group: Optional[str] = None,
            metrics_tracker: Optional[MetricsTracker] = None
    ) -> None:
        """
        Initialize a WandbLogger.
        NOTE: The metric_tracker composition functionality is legacy (and can be removed later)
        but I am keeping the logger itself for compatibility with the BayesianOptimizer class.
        For actual training/not hyperparameter tuning, use the WandbCallback instead.

        :param enable: If False, disables all logging.
        :param project: Name of the W&B project.
        :param entity: W&B entity (user or team).
        :param config: Optional run configuration dictionary.
        :param name: Optional name of the run.
        :param group: Group of the run.
        :param metrics_tracker: Optional MetricsTracker instance for keeping track of metrics across runs.
        """
        self.enabled = enable
        self.project = project
        self.entity = entity
        self.config = config
        self.name = name
        self.group = group
        self.metrics_tracker = metrics_tracker
        self.run = None
        self.runs = 0
        # self.start()

    def start(self) -> None:
        """
        Start a new W&B run with timestamped name.
        """
        # Get current date and time
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.enabled:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                name=self.name + "-" + formatted_time,
                group=self.group
            )
        self.runs += 1

    def add_metrics_tracker(self, metrics_tracker: MetricsTracker) -> None:
        """
        Attach a new metrics tracker.

        :param metrics_tracker: MetricsTracker instance.
        """
        self.metrics_tracker = metrics_tracker

    def enable(self) -> None:
        """
        Enable logging.
        """
        self.enabled = True

    def disable(self) -> None:
        """
        Disable logging.
        """
        self.enabled = False

    def has_metric_tracker(self) -> bool:
        """
        Check if a metrics tracker is attached.

        :return: True if a metrics tracker is present.
        """
        return self.metrics_tracker is not None

    def log(self, metrics: Dict[str, Any], agent_id: Optional[str] = None, episode: Optional[int] = None) -> None:
        """
        Log metrics to W&B and optionally record them with the tracker.

        :param metrics: Dictionary of metrics to log.
        :param agent_id: Optional identifier for the agent.
        :param episode: Optional episode number.
        """
        if self.enabled:
            wandb.log(metrics)

        if self.metrics_tracker:
            for key, value in metrics.items():
                self.metrics_tracker.record_metric(key, agent_id, episode, value)

    def save(self, filepath: str) -> None:
        """
        Save a file artifact to W&B.

        :param filepath: Path to the file.
        """
        if self.enabled:
            wandb.save(filepath)

    def log_metric_tracker_state(self, num_episodes: int, export_metrics: bool = True) -> None:
        """
        Log saved metrics and plots via W&B.

        :param num_episodes: Total number of episodes for plotting.
        :param export_metrics: If True, save metrics to disk and upload to W&B.
        """
        # TODO: Remove requirement to pass num_episodes its kinda annoying ngl
        # TODO: This method is kinda messy in general
        if self.metrics_tracker and self.enabled:
            self.metrics_tracker.plot_all_metrics(num_episodes)
            save_path = self.metrics_tracker.save_path
            metrics_path = os.path.join(save_path, "metrics.json")   # Also kind of sketchy
            for metric_name in self.metrics_tracker.metrics_history.keys():
                self.log_image(metric_name, str(os.path.join(save_path, metric_name)) + ".png")
                self.save(str(os.path.join(save_path, metric_name)) + ".svg")

            if export_metrics:
                self.metrics_tracker.export_metrics(metrics_path)
                self.save(metrics_path)

    def log_image(self, key: str, image_path: str) -> None:
        """
        Log an image to W&B.

        :param key: Name to use in W&B logs.
        :param image_path: Path to the image file.
        """
        if self.enabled:
            wandb.log({key: wandb.Image(image_path)})

    def finish(self) -> None:
        """
        Finish the current W&B run.
        """
        if self.enabled:
            wandb.finish()
