import wandb
import os
from typing import Optional, Dict, Any
from datetime import datetime
from src.metrics.metrictracker import MetricsTracker


class WandbLogger:
    def __init__(
            self,
            enable: bool = True,
            project: Optional[str] = None,
            entity: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            metrics_tracker: Optional[MetricsTracker] = None
    ) -> None:
        self.enabled = enable
        self.project = project
        self.entity = entity
        self.config = config
        self.name = name
        self.metrics_tracker = metrics_tracker

        # Get current date and time
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=name + "-" + formatted_time
            )

    def add_metrics_tracker(self, metrics_tracker: MetricsTracker) -> None:
        self.metrics_tracker = metrics_tracker

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def has_metric_tracker(self) -> bool:
        return self.metrics_tracker is not None

    def log(self, metrics: Dict[str, Any], agent_id: Optional[str] = None, episode: Optional[int] = None) -> None:
        if self.enabled:
            wandb.log(metrics)

        if self.metrics_tracker:
            for key, value in metrics.items():
                self.metrics_tracker.record_metric(key, agent_id, episode, value)

    def save(self, filepath: str) -> None:
        if self.enabled:
            wandb.save(filepath)

    def log_metric_tracker_state(self, num_episodes: int, export_metrics: bool = True) -> None:
        # TODO: Remove requirement to pass num_episodes its kinda annoying ngl
        # TODO: This method is kinda messy in general
        if self.metrics_tracker and self.enabled:
            self.metrics_tracker.plot_all_metrics(num_episodes)
            save_path = self.metrics_tracker.save_path
            metrics_path = os.path.join(save_path, "metrics.json")   # Also kind off sketchy
            for metric_name in self.metrics_tracker.metrics_history.keys():
                self.log_image(metric_name, str(os.path.join(save_path, metric_name)) + ".png")
                self.save(str(os.path.join(save_path, metric_name)) + ".svg")

            if export_metrics:
                self.metrics_tracker.export_metrics(metrics_path)
                self.save(metrics_path)

    def log_image(self, key: str, image_path: str) -> None:
        if self.enabled:
            wandb.log({key: wandb.Image(image_path)})

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()
