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
    Standalone WandB callback with parallel evaluation environment.
    
    Features:
    - Logs average step reward for training environment (handles parallel envs)
    - Runs evaluation environment in parallel, logging step-by-step metrics
    - Tracks episode returns for both train and eval separately
    - Supports WandB groups for organizing runs
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
        log_step_frequency: int = 1,  # Log every N steps (1 = every step)
        
        # Evaluation environment
        eval_env_manager: Optional[Any] = None,  # Separate EnvManager for evaluation
        eval_log_prefix: str = "eval_",  # Prefix for eval metrics in wandb
        eval_sync_frequency: int = 1,  # Sync eval every N training steps
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
        self.eval_sync_frequency = eval_sync_frequency
        self.do_evaluation = eval_env_manager is not None
        
        # Training environment state
        self.n_env = None
        self.train_episode_returns = None  # Ongoing returns for each train env
        self.train_completed_returns = []  # List of completed train episode returns
        self.train_episode_lengths = []  # List of completed train episode lengths
        self.train_episodes_finished = 0
        self.total_train_timesteps = 0
        
        # Evaluation environment state
        self.eval_episode_returns = None  # Ongoing returns for each eval env
        self.eval_completed_returns = []  # List of completed eval episode returns
        self.eval_episode_lengths = []  # List of completed eval episode lengths
        self.eval_episodes_finished = 0
        self.total_eval_timesteps = 0
        self.eval_step_counter = 0
        
        # For step-level statistics
        self.train_step_rewards_history = []
        self.eval_step_rewards_history = []
        self.avg_step_reward_window = 1000
        
        # Agent reference and synchronization
        self.agent = None
        self.run = None
        self.eval_running = False
        self.eval_thread = None
        logger.setLevel(verbose)

    def init_callback(self, data: SimulatorRLData):
        """
        Initialize WandB run and internal state for both train and eval.
        """
        super().init_callback(data)
        
        # Get number of parallel environments from simulator
        self.n_env = data.n_env
        self.train_episode_returns = np.zeros(self.n_env, dtype=np.float32)
        
        # Store agent reference
        self.agent = data.agent
        
        # Initialize eval environment state if needed
        if self.do_evaluation:
            self.eval_episode_returns = np.zeros(self.eval_env_manager.n_envs, dtype=np.float32)
            logger.info(f"Evaluation enabled with {self.eval_env_manager.n_envs} environment(s)")
        
        # Initialize WandB run with group
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
            f"(project={self.project}, group={self.group}, run={self.run_name})"
        )
        
        # Start evaluation thread if needed
        if self.do_evaluation:
            self._start_evaluation_thread()

    def _start_evaluation_thread(self):
        """Start a separate thread to run the evaluation environment."""
        if self.eval_running:
            return
            
        self.eval_running = True
        self.eval_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.eval_thread.start()
        logger.info("Evaluation thread started")

    def _evaluation_loop(self):
        """Main loop for evaluation environment."""
        if not self.eval_env_manager or not self.agent:
            return
            
        # Initialize eval environment
        eval_obs = self.eval_env_manager.reset()
        eval_dones = np.zeros(self.eval_env_manager.n_envs, dtype=bool)
        
        while self.eval_running:
            try:
                # Get actions from agent (deterministic for evaluation)
                eval_actions = self.agent.choose_action(eval_obs, deterministic=True)
                
                # Step the evaluation environment
                next_eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_env_manager.step(eval_actions)
                
                # Update eval episode returns
                self.eval_episode_returns += eval_rewards
                
                # Handle eval episode completion
                for i in range(self.eval_env_manager.n_envs):
                    if eval_dones[i]:
                        self._handle_eval_episode_completion(i)
                
                # Log eval step metrics
                self._log_eval_step(eval_rewards, eval_dones)
                
                # Update for next step
                eval_obs = next_eval_obs
                self.total_eval_timesteps += self.eval_env_manager.n_envs
                
                # Control evaluation speed (optional)
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
                
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
                break

    def _handle_eval_episode_completion(self, env_idx: int):
        """Handle completion of an episode in evaluation environment."""
        episode_return = self.eval_episode_returns[env_idx]
        self.eval_completed_returns.append(episode_return)
        self.eval_episodes_finished += 1
        
        # Calculate episode length (approximate)
        episode_length = self.total_eval_timesteps // max(1, self.eval_episodes_finished)
        self.eval_episode_lengths.append(episode_length)
        
        # Log eval episode completion
        logger.debug(
            f"Eval episode {self.eval_episodes_finished} finished in env {env_idx} "
            f"with return {episode_return:.2f}"
        )
        
        # Reset this environment's return tracker
        self.eval_episode_returns[env_idx] = 0.0

    def _log_eval_step(self, rewards, dones):
        """Log evaluation step metrics."""
        if not self.do_evaluation:
            return
            
        self.eval_step_counter += 1
        
        # Calculate eval step statistics
        if isinstance(rewards, np.ndarray):
            avg_eval_reward = np.mean(rewards)
            max_eval_reward = np.max(rewards)
            min_eval_reward = np.min(rewards)
            std_eval_reward = np.std(rewards)
        else:
            avg_eval_reward = rewards
            max_eval_reward = rewards
            min_eval_reward = rewards
            std_eval_reward = 0.0
        
        # Log eval metrics
        eval_metrics = {
            f"{self.eval_log_prefix}timestep": self.total_eval_timesteps,
            f"{self.eval_log_prefix}avg_step_reward": avg_eval_reward,
            f"{self.eval_log_prefix}max_step_reward": max_eval_reward,
            f"{self.eval_log_prefix}min_step_reward": min_eval_reward,
            f"{self.eval_log_prefix}std_step_reward": std_eval_reward,
            f"{self.eval_log_prefix}episodes_finished": self.eval_episodes_finished,
        }
        
        # Add ongoing returns for eval environments
        for i in range(min(self.eval_env_manager.n_envs, 3)):
            eval_metrics[f"{self.eval_log_prefix}env_{i}_ongoing_return"] = self.eval_episode_returns[i]
        
        # Track for moving average
        self.eval_step_rewards_history.append(avg_eval_reward)
        if len(self.eval_step_rewards_history) > self.avg_step_reward_window:
            self.eval_step_rewards_history.pop(0)
        
        # Log to wandb (in the main thread via thread-safe logging)
        if self.run:
            # Note: wandb.log is thread-safe
            wandb.log(eval_metrics)

    def on_step(self, action, reward, next_obs, done) -> bool:
        """
        Handle training step-level logging.
        """
        super().on_step(action, reward, next_obs, done)
        
        self.total_train_timesteps += 1
        
        # Handle multiple training environments
        if isinstance(reward, np.ndarray):
            # Multiple parallel environments
            avg_train_reward = np.mean(reward)
            max_train_reward = np.max(reward)
            min_train_reward = np.min(reward)
            std_train_reward = np.std(reward)
            
            # Update training episode returns
            self.train_episode_returns += reward
            
            # Handle training episode completion
            for i in range(self.n_env):
                if done[i]:
                    self._handle_train_episode_completion(i)
        else:
            # Single training environment
            avg_train_reward = reward
            max_train_reward = reward
            min_train_reward = reward
            std_train_reward = 0.0
            
            # Update episode return
            self.train_episode_returns[0] += reward if self.n_env == 1 else 0
            
            # Handle episode completion
            if done:
                self._handle_train_episode_completion(0)
        
        # Log training step-level metrics
        if self.total_train_timesteps % self.log_step_frequency == 0:
            self._log_train_step(avg_train_reward, max_train_reward, 
                               min_train_reward, std_train_reward)
        
        # Track for moving average
        self.train_step_rewards_history.append(avg_train_reward)
        if len(self.train_step_rewards_history) > self.avg_step_reward_window:
            self.train_step_rewards_history.pop(0)
        
        return True
    
    def _handle_train_episode_completion(self, env_idx: int):
        """Handle completion of an episode in training environment."""
        episode_return = self.train_episode_returns[env_idx]
        self.train_completed_returns.append(episode_return)
        self.train_episodes_finished += 1
        
        # Calculate episode length
        episode_length = self.total_train_timesteps // max(1, self.train_episodes_finished)
        self.train_episode_lengths.append(episode_length)
        
        # Log training episode metrics
        train_episode_metrics = {
            "train_episode": self.train_episodes_finished,
            "train_episode_return": episode_return,
            "train_avg_episode_return": np.mean(self.train_completed_returns[-100:]),
            "train_episode_length": episode_length,
            "train_env_idx": env_idx,
            "train_timestep": self.total_train_timesteps,
        }
        wandb.log(train_episode_metrics)
        
        # Also log eval episode metrics if available (for comparison)
        if self.eval_completed_returns:
            train_episode_metrics["eval_vs_train_ratio"] = (
                np.mean(self.eval_completed_returns[-10:]) / 
                np.mean(self.train_completed_returns[-10:]) 
                if self.train_completed_returns[-10:] else 0
            )
        
        logger.debug(
            f"Train episode {self.train_episodes_finished} finished in env {env_idx} "
            f"with return {episode_return:.2f}"
        )
        
        # Reset this environment's return tracker
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
        
        # Add ongoing returns for training environments
        for i in range(min(self.n_env, 5)):
            step_metrics[f"train_env_{i}_ongoing_return"] = self.train_episode_returns[i]
        
        # Add comparison with eval if available
        if self.eval_step_rewards_history:
            step_metrics["train_vs_eval_reward_ratio"] = (
                avg_reward / np.mean(self.eval_step_rewards_history[-100:]) 
                if self.eval_step_rewards_history else 1.0
            )
        
        wandb.log(step_metrics)

    def on_learn(self, learning_info: Dict[str, Any]):
        """
        Log learning metrics from agent updates.
        """
        super().on_learn(learning_info)
        
        if learning_info:
            # Add timestep to learning info
            learning_info_with_step = learning_info.copy()
            learning_info_with_step["train_timestep"] = self.total_train_timesteps
            wandb.log(learning_info_with_step)

    def _log_plots(self):
        """Log plot files to WandB."""
        if not self.plot_dir or not self.plot_dir.exists():
            return

        artifact = wandb.Artifact(f"{self.run_name}_plots", type="plots")
        for ext in ("*.png", "*.svg", "*.pdf"):
            for file in self.plot_dir.glob(ext):
                wandb.log({f"plots/{file.stem}": wandb.Image(str(file))})
                artifact.add_file(str(file))
                logger.debug(f"Logged plot: {file.name}")
        
        if artifact.files:
            self.run.log_artifact(artifact)

    def on_training_end(self):
        """Final logging and cleanup."""
        super().on_training_end()
        
        # Stop evaluation thread
        self.eval_running = False
        if self.eval_thread:
            self.eval_thread.join(timeout=2.0)
            logger.info("Evaluation thread stopped")
        
        if self.do_evaluation and self.eval_env_manager:
            self.eval_env_manager.close()
        
        # Log final statistics
        self._log_final_stats()
        
        # Log any plots
        self._log_plots()
        
        # Create and log summary CSVs
        self._log_summary_csv()
        
        # Finish WandB run
        wandb.finish()
        logger.info("WandbStepCallback training ended and WandB run closed.")

    def _log_final_stats(self):
        """Log final statistics for both train and eval."""
        final_stats = {}
        
        # Training statistics
        if self.train_completed_returns:
            final_stats.update({
                "final/train_avg_episode_return": np.mean(self.train_completed_returns),
                "final/train_std_episode_return": np.std(self.train_completed_returns),
                "final/train_min_episode_return": np.min(self.train_completed_returns),
                "final/train_max_episode_return": np.max(self.train_completed_returns),
                "final/train_total_episodes": len(self.train_completed_returns),
                "final/train_total_timesteps": self.total_train_timesteps,
                "final/train_avg_step_reward": np.mean(self.train_step_rewards_history) 
                if self.train_step_rewards_history else 0,
            })
        
        # Evaluation statistics
        if self.do_evaluation and self.eval_completed_returns:
            final_stats.update({
                "final/eval_avg_episode_return": np.mean(self.eval_completed_returns),
                "final/eval_std_episode_return": np.std(self.eval_completed_returns),
                "final/eval_min_episode_return": np.min(self.eval_completed_returns),
                "final/eval_max_episode_return": np.max(self.eval_completed_returns),
                "final/eval_total_episodes": len(self.eval_completed_returns),
                "final/eval_total_timesteps": self.total_eval_timesteps,
                "final/eval_avg_step_reward": np.mean(self.eval_step_rewards_history) 
                if self.eval_step_rewards_history else 0,
                
                # Comparison metrics
                "final/train_eval_return_ratio": (
                    np.mean(self.train_completed_returns) / np.mean(self.eval_completed_returns) 
                    if self.eval_completed_returns else 0
                ),
            })
        
        if final_stats:
            wandb.log(final_stats)

    def _log_summary_csv(self):
        """Create and log summary CSVs for both train and eval."""
        import pandas as pd
        import tempfile
        
        # Training summary
        if self.train_completed_returns:
            train_data = {
                "episode": list(range(1, len(self.train_completed_returns) + 1)),
                "return": self.train_completed_returns,
                "length": self.train_episode_lengths[:len(self.train_completed_returns)],
            }
            
            df_train = pd.DataFrame(train_data)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df_train.to_csv(f.name, index=False)
                artifact = wandb.Artifact(f"{self.run_name}_train_summary", type="metrics")
                artifact.add_file(f.name, name="train_episode_summary.csv")
                self.run.log_artifact(artifact)
        
        # Evaluation summary
        if self.do_evaluation and self.eval_completed_returns:
            eval_data = {
                "episode": list(range(1, len(self.eval_completed_returns) + 1)),
                "return": self.eval_completed_returns,
                "length": self.eval_episode_lengths[:len(self.eval_completed_returns)],
            }
            
            df_eval = pd.DataFrame(eval_data)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df_eval.to_csv(f.name, index=False)
                artifact = wandb.Artifact(f"{self.run_name}_eval_summary", type="metrics")
                artifact.add_file(f.name, name="eval_episode_summary.csv")
                self.run.log_artifact(artifact)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for external use."""
        stats = {
            "train": {
                "total_timesteps": self.total_train_timesteps,
                "episodes_finished": self.train_episodes_finished,
                "avg_episode_return": np.mean(self.train_completed_returns) if self.train_completed_returns else 0,
                "ongoing_returns": self.train_episode_returns.tolist() if self.train_episode_returns is not None else [],
            }
        }
        
        if self.do_evaluation:
            stats["eval"] = {
                "total_timesteps": self.total_eval_timesteps,
                "episodes_finished": self.eval_episodes_finished,
                "avg_episode_return": np.mean(self.eval_completed_returns) if self.eval_completed_returns else 0,
                "ongoing_returns": self.eval_episode_returns.tolist() if self.eval_episode_returns is not None else [],
            }
        
        return stats
