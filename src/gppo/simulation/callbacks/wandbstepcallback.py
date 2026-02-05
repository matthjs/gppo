import logging
import pathlib
import wandb
import numpy as np
import threading
import time
from typing import Optional, Dict, Any, List, Union
from gppo.simulation.callbacks.abstractcallback import AbstractCallback
from gppo.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)

class WandbStepCallback(AbstractCallback):
    """
    WandB callback with periodic evaluation.
    
    Features:
    - Logs training metrics step-by-step
    - Runs evaluation periodically (every N training steps)
    - No background thread - evaluation is synchronous with training
    """

    def __init__(
        self,
        project: str,
        entity: str = None,
        config: Dict[str, Any] = None,
        run_name: str = None,
        group: str = None,
        tags: List[str] = None,
        verbose: int = logging.INFO,
        plot_dir: str = None,
        log_step_frequency: int = 1,
        
        # Evaluation settings
        eval_env_manager: Optional[Any] = None,
        eval_log_prefix: str = "eval_",
        eval_frequency: int = 1000,  # Evaluate every N training steps
        eval_episodes: int = 10,  # Number of episodes to run per evaluation
    ):
        super().__init__()
        self.project = project
        self.run_name = run_name
        self.group = group
        self.entity = entity
        self.tags = tags or []
        self.config = config or {}
        self.plot_dir = pathlib.Path(plot_dir) if plot_dir else None
        self.log_step_frequency = log_step_frequency
        
        # Evaluation settings
        self.eval_env_manager = eval_env_manager
        self.eval_log_prefix = eval_log_prefix
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.do_evaluation = eval_env_manager is not None
        
        # Training state
        self.n_env = None
        self.train_episode_returns = None
        self.train_completed_returns = []
        self.train_episode_lengths = []
        self.train_episodes_finished = 0
        self.total_train_timesteps = 0
        
        # Evaluation state
        self.eval_completed_returns = []
        self.eval_episode_lengths = []
        self.total_eval_episodes = 0
        self.last_eval_step = 0
        
        # Statistics
        self.train_step_rewards_history = []
        self.avg_step_reward_window = 1000
        
        self.agent = None
        self.run = None
        logger.setLevel(verbose)

    def init_callback(self, data: SimulatorRLData):
        """Initialize WandB run and internal state."""
        super().init_callback(data)
        
        self.n_env = data.n_env
        self.train_episode_returns = np.zeros(self.n_env, dtype=np.float32)
        self.agent = data.agent
        
        if self.do_evaluation:
            logger.info(f"Evaluation enabled: every {self.eval_frequency} steps, {self.eval_episodes} episodes")
        
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            group=self.group,
            tags=self.tags,
            config=self.config,
            reinit=True,
        )
        
        logger.info(
            f"WandbStepCallback initialized with {self.n_env} training environments "
            f"(project={self.project}, group={self.group})"
        )

    def on_step(self, action, reward, next_obs, done) -> bool:
        """Handle training step-level logging and periodic evaluation."""
        super().on_step(action, reward, next_obs, done)
        
        self.total_train_timesteps += 1
        
        # Handle training metrics
        if isinstance(reward, np.ndarray):
            avg_train_reward = np.mean(reward)
            max_train_reward = np.max(reward)
            min_train_reward = np.min(reward)
            std_train_reward = np.std(reward)
            
            self.train_episode_returns += reward
            
            for i in range(self.n_env):
                if done[i]:
                    self._handle_train_episode_completion(i)
        else:
            avg_train_reward = reward
            max_train_reward = reward
            min_train_reward = reward
            std_train_reward = 0.0
            
            self.train_episode_returns[0] += reward if self.n_env == 1 else 0
            
            if done:
                self._handle_train_episode_completion(0)
        
        # Log training step metrics
        if self.total_train_timesteps % self.log_step_frequency == 0:
            self._log_train_step(avg_train_reward, max_train_reward, 
                               min_train_reward, std_train_reward)
        
        # Track for moving average
        self.train_step_rewards_history.append(avg_train_reward)
        if len(self.train_step_rewards_history) > self.avg_step_reward_window:
            self.train_step_rewards_history.pop(0)
        
        # Run periodic evaluation
        if (self.do_evaluation and 
            self.total_train_timesteps - self.last_eval_step >= self.eval_frequency):
            self._run_evaluation()
            self.last_eval_step = self.total_train_timesteps
        
        return True

    def _run_evaluation(self):
        """Run evaluation for specified number of episodes."""
        logger.info(f"Running evaluation at step {self.total_train_timesteps}...")
        
        eval_episode_returns = []
        eval_episode_lengths = []
        
        for ep in range(self.eval_episodes):
            obs = self.eval_env_manager.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                # Get action from agent (deterministic for eval)
                action = self.agent.choose_action(obs)
                
                # Step environment
                obs, reward, done, info = self.eval_env_manager.step(action)
                
                episode_return += reward if not isinstance(reward, np.ndarray) else np.sum(reward)
                episode_length += 1
                
                # Handle vectorized env
                if isinstance(done, np.ndarray):
                    done = done[0]
            
            eval_episode_returns.append(episode_return)
            eval_episode_lengths.append(episode_length)
            self.total_eval_episodes += 1
        
        # Store results
        self.eval_completed_returns.extend(eval_episode_returns)
        self.eval_episode_lengths.extend(eval_episode_lengths)
        
        # Log evaluation metrics
        eval_metrics = {
            f"{self.eval_log_prefix}timestep": self.total_train_timesteps,
            f"{self.eval_log_prefix}mean_return": np.mean(eval_episode_returns),
            f"{self.eval_log_prefix}std_return": np.std(eval_episode_returns),
            f"{self.eval_log_prefix}min_return": np.min(eval_episode_returns),
            f"{self.eval_log_prefix}max_return": np.max(eval_episode_returns),
            f"{self.eval_log_prefix}mean_length": np.mean(eval_episode_lengths),
            f"{self.eval_log_prefix}total_episodes": self.total_eval_episodes,
        }
        
        wandb.log(eval_metrics)
        
        logger.info(
            f"Evaluation complete: mean_return={np.mean(eval_episode_returns):.2f}, "
            f"episodes={len(eval_episode_returns)}"
        )

    def _handle_train_episode_completion(self, env_idx: int):
        """Handle completion of an episode in training environment."""
        episode_return = self.train_episode_returns[env_idx]
        self.train_completed_returns.append(episode_return)
        self.train_episodes_finished += 1
        
        episode_length = self.total_train_timesteps // max(1, self.train_episodes_finished)
        self.train_episode_lengths.append(episode_length)
        
        train_episode_metrics = {
            "train_episode": self.train_episodes_finished,
            "train_episode_return": episode_return,
            "train_avg_episode_return": np.mean(self.train_completed_returns[-100:]),
            "train_episode_length": episode_length,
            "train_timestep": self.total_train_timesteps,
        }
        wandb.log(train_episode_metrics)
        
        logger.debug(
            f"Train episode {self.train_episodes_finished} finished in env {env_idx} "
            f"with return {episode_return:.2f}"
        )
        
        self.train_episode_returns[env_idx] = 0.0

    def _log_train_step(self, avg_reward, max_reward, min_reward, std_reward):
        """Log training step metrics."""
        step_metrics = {
            "train_timestep": self.total_train_timesteps,
            "train_avg_step_reward": avg_reward,
            "train_max_step_reward": max_reward,
            "train_min_step_reward": min_reward,
            "train_std_step_reward": std_reward,
            "train_episodes_finished": self.train_episodes_finished,
        }
        
        for i in range(min(self.n_env, 5)):
            step_metrics[f"train_env_{i}_ongoing_return"] = self.train_episode_returns[i]
        
        wandb.log(step_metrics)

    def on_learn(self, learning_info: Dict[str, Any]):
        """Log learning metrics from agent updates."""
        super().on_learn(learning_info)
        
        if learning_info:
            learning_info_with_step = learning_info.copy()
            learning_info_with_step["train_timestep"] = self.total_train_timesteps
            wandb.log(learning_info_with_step)

    def on_training_end(self):
        """Final logging and cleanup."""
        super().on_training_end()
        
        # Run final evaluation
        if self.do_evaluation:
            logger.info("Running final evaluation...")
            self._run_evaluation()
            self.eval_env_manager.close()
        
        self._log_final_stats()
        self._log_plots()
        self._log_summary_csv()
        
        wandb.finish()
        logger.info("WandbStepCallback training ended and WandB run closed.")

    def _log_final_stats(self):
        """Log final statistics."""
        final_stats = {}
        
        if self.train_completed_returns:
            final_stats.update({
                "final/train_avg_episode_return": np.mean(self.train_completed_returns),
                "final/train_std_episode_return": np.std(self.train_completed_returns),
                "final/train_total_episodes": len(self.train_completed_returns),
                "final/train_total_timesteps": self.total_train_timesteps,
            })
        
        if self.eval_completed_returns:
            final_stats.update({
                "final/eval_avg_episode_return": np.mean(self.eval_completed_returns),
                "final/eval_std_episode_return": np.std(self.eval_completed_returns),
                "final/eval_total_episodes": self.total_eval_episodes,
            })
        
        if final_stats:
            wandb.log(final_stats)

    def _log_plots(self):
        """Log plot files to WandB."""
        if not self.plot_dir or not self.plot_dir.exists():
            return

        for ext in ("*.png", "*.svg", "*.pdf"):
            for file in self.plot_dir.glob(ext):
                wandb.log({f"plots/{file.stem}": wandb.Image(str(file))})

    def _log_summary_csv(self):
        """Create and log summary CSVs."""
        import pandas as pd
        import tempfile
        
        if self.train_completed_returns:
            df_train = pd.DataFrame({
                "episode": list(range(1, len(self.train_completed_returns) + 1)),
                "return": self.train_completed_returns,
                "length": self.train_episode_lengths[:len(self.train_completed_returns)],
            })
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df_train.to_csv(f.name, index=False)
                artifact = wandb.Artifact(f"{self.run_name}_train_summary", type="metrics")
                artifact.add_file(f.name, name="train_episode_summary.csv")
                self.run.log_artifact(artifact)
