import wandb
from typing import Optional, Dict, Any

from src.metrics.metrictracker import MetricsTracker


class WandbLogger:
    def __init__(
            self,
            enable: bool = True,
            project: Optional[str] = None,
            entity: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            run_name: Optional[str] = None,
            metrics_tracker: Optional[MetricsTracker] = None
    ) -> None:
        self.enabled = True
        self.project = project
        self.entity = entity
        self.config = config
        self.run_name = run_name
        self.metrics_tracker = metrics_tracker

        if self.enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=run_name
            )

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

    def log_metric_tracker_state(self) -> None:
        pass

    def log_image(self, key: str, image_path: str) -> None:
        if self.enabled:
            wandb.log({key: wandb.Image(image_path)})

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()
