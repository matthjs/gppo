"""
Taken from: https://github.com/matthjs/xai-gp/blob/main/xai_gp/hyperparam_tuning/bayesianoptimizer.py
with slight adjustments to ensure compatibility with RL Agents.
"""
import json
import os
from typing import Callable, Tuple, List, Union
from ax import Experiment
from ax.service.ax_client import AxClient
import torch
from typing import Dict, Any, Optional
from ax.service.utils.instantiation import ObjectiveProperties
from src.agents.agent import Agent
from src.util.wandblogger import WandbLogger
import warnings

warnings.filterwarnings(
    "once",
    category=RuntimeWarning,
    message=r"If using a 2-dim `batch_initial_conditions`.*"
)


class BayesianOptimizer:
    def __init__(
            self,
            search_space: List[dict],
            model_factory: Callable[[Dict[str, Any]], Union[torch.nn.Module, Agent]],
            train_fn: Callable[[Union[torch.nn.Module, Agent], Dict[str, Any]], Dict[str, float]],
            eval_fn: Callable[[Union[torch.nn.Module, Agent], Dict[str, Any]], Dict[str, float]],
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tracking_metrics: Optional[Tuple[str]] = None,
            objective_name: str = "loss",
            minimize: bool = True,
            wandb_logger: Optional[WandbLogger] = None,
            save_path: Optional[str] = None,
    ):
        """
        Generic Bayesian optimization wrapper for various models.

        :param search_space: Ax parameter search space definition
        :param model_factory: Function that creates model from parameters
            A function that takes in a dictionary of hyperparameters and returns a nn.Module instance.
        :param train_fn: Function that trains model and returns loss
            The train function expects a model (nn.Module) and a dictionary of parameters
            and returns a float value (e.g., the training loss).
        :param eval_fn: Function that evaluates model and returns metrics
            The evaluation function expects a model (nn.Module) and returns a dictionary of metrics.
        :param device: Target device for computation
        :param tracking_metrics: Metrics to track (defaults to eval_fn keys)

        Bayesian optimization loop:
        1: Initialize with a set of random hyperparameters and evaluate objective f(x) at these points.
        2: Fit the surrogate model (e.g., Gaussian process) to the observed data {x_i, f(x_i)}.
        3: A promising new set of hyperparameters x_new are queried that maximize the acquisition function
        (a lot of them involve the uncertainty variance of the GP predictive posterior distribution)
        4: Evaluate f(x_new)
        5: Augment dataset with {(x_new, f(x_new)}.
        6: Goto 1. until stopping criterion is met.
        """
        self.ax_client = AxClient(torch_device=device)
        self.device = device
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.tracking_metrics = tracking_metrics

        self.objective_name = objective_name
        self.minimize = minimize
        self.save_path = save_path

        # Initialize hyperparameter tuning experiment. We want to find the optimal set of
        # hyperparameters such that an objective is minimized (or maximized)
        self.ax_client.create_experiment(
            name="bayesian_optimization",
            parameters=search_space,
            objectives={self.objective_name: ObjectiveProperties(minimize=self.minimize)},
        )

        self.logger = wandb_logger

    def run_trial(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single optimization trial"""
        model = self.model_factory(params)
        if hasattr(model, "to"):
            model = model.to(self.device)
        metrics = self.train_fn(model, params)
        metrics.update(self.eval_fn(model, params))

        # Loop and print all metrics
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")
        print("---")

        return metrics

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """Main optimization loop"""
        for _ in range(n_trials):
            # Get suggested hyperparameter values.
            params, trial_idx = self.ax_client.get_next_trial()
            metrics = self.run_trial(params)

            # Report primary metric to Ax
            self.ax_client.complete_trial(
                trial_index=trial_idx,
                raw_data=metrics[self.objective_name]
            )

        best = self.get_best_parameters()
        if self.logger and self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, "best_hyperparams.json")
            with open(path, "w") as f:
                json.dump(best, f, indent=4)
            self.logger.save(path)
            print(f"Saved best hyperparameters to {path}")
        return best

    def get_best_parameters(self) -> Dict[str, Any]:
        """Return best parameters found"""
        return self.ax_client.get_best_parameters()[0]

    @property
    def experiment(self) -> Experiment:
        """Access underlying Ax experiment object."""
        return self.ax_client.experiment

    def save_state(self, filename: str) -> None:
        """Save experiment state to file."""
        self.ax_client.save_to_json_file(filename)

    @classmethod
    def load_state(cls, filename: str, train_data: Dict[str, torch.Tensor]) -> "BayesianOptimizer":
        """Load existing experiment from file."""
        new_instance = cls.__new__(cls)
        new_instance.ax_client = AxClient.load_from_json_file(filename)
        new_instance.train_data = train_data
        return new_instance
