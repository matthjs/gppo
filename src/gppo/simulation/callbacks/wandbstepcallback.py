import logging
import pathlib
import wandb
import numpy as np
from typing import Optional, Dict, Any, List, Union
from gppo.simulation.callbacks.abstractcallback import AbstractCallback
from gppo.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)


class WandbStepCallback(AbstractCallback):
    """
    Standalone WandB callback with evaluation support.
    
    Features:
    - Logs average step reward (handles parallel environments)
    - Tracks episode returns for each environment separately
    - Logs episode-level metrics when episodes complete
    - Supports periodic evaluation on separate eval environment
    - Supports WandB groups for organizing runs
    - Optional step-level logging frequency to avoid flooding WandB
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
        log_step_rewards: bool = True,  # Whether to log step rewards
        log_episode_returns: bool = True,  # Whether to log episode returns
        
        # Evaluation parameters
        eval_env_manager: Optional[Any] = None,  # EnvManager for evaluation
        eval_frequency: int = 0,  # Evaluate every N episodes (0 = no evaluation)
        eval_episodes: int = 5,  # Number of episodes per evaluation
        eval_prefix: str = "eval/",  # Prefix for eval metrics in wandb
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
        self.log_step_rewards = log_step_rewards
        self.log_episode_returns = log_episode_returns
        
        # Evaluation settings
        self.eval_env_manager = eval_env_manager
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.eval_prefix = eval_prefix
        self.do_evaluation = eval_env_manager is not None and eval_frequency > 0
        
        # Internal state tracking
        self.n_env = None
        self.episode_returns = None  # Ongoing returns for each env
        self.completed_returns = []  # List of completed episode returns
        self.episode_lengths = []  # List of completed episode lengths
        self.episodes_finished = 0
        self.total_timesteps = 0
        
        # For step-level statistics
        self.step_rewards_history = []
        self.avg_step_reward_window = 1000  # Window for moving average
        
        # Evaluation tracking
        self.eval_returns_history = []  # List of eval returns over time
        self.eval_episodes_history = []  # Corresponding episode numbers
        self.last_eval_episode = 0
        
        # Agent reference for evaluation
        self.agent = None
        self.run = None
        logger.setLevel(verbose)

    def init_callback(self, data: SimulatorRLData):
        """
        Initialize WandB run and internal state.
        """
        super().init_callback(data)
        
        # Get number of parallel environments from simulator
        self.n_env = data.n_env
        self.episode_returns = np.zeros(self.n_env, dtype=np.float32)
        
        # Store agent reference for evaluation
        self.agent = data.agent
        
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
            f"WandbStepCallback initialized with {self.n_env} environments "
            f"(project={self.project}, group={self.group}, run={self.run_name})"
        )
        
        if self.do_evaluation:
            logger.info(
                f"Evaluation enabled: every {self.eval_frequency} episodes, "
                f"{self.eval_episodes} episodes per evaluation"
            )

    def on_step(self, action, reward, next_obs, done) -> bool:
        """
        Handle step-level logging.
        """
        super().on_step(action, reward, next_obs, done)
        
        self.total_timesteps += 1
        
        # Handle multiple environments
        if isinstance(reward, np.ndarray):
            # Multiple parallel environments
            avg_reward = np.mean(reward)
            max_reward = np.max(reward)
            min_reward = np.min(reward)
            std_reward = np.std(reward)
            
            # Update episode returns for each environment
            self.episode_returns += reward
            
            # Handle episode completion
            for i in range(self.n_env):
                if done[i]:
                    self._handle_episode_completion(i)
        else:
            # Single environment
            avg_reward = reward
            max_reward = reward
            min_reward = reward
            std_reward = 0.0
            
            # Update episode return
            self.episode_returns[0] += reward if self.n_env == 1 else 0
            
            # Handle episode completion
            if done:
                self._handle_episode_completion(0)
        
        # Log step-level metrics with specified frequency
        if self.log_step_rewards and (self.total_timesteps % self.log_step_frequency == 0):
            step_metrics = {
                "timestep": self.total_timesteps,
                "avg_step_reward": avg_reward,
                "max_step_reward": max_reward,
                "min_step_reward": min_reward,
                "std_step_reward": std_reward,
                "episodes_finished": self.episodes_finished,
            }
            
            # Add ongoing returns for each environment
            for i in range(min(self.n_env, 5)):  # Limit to first 5 envs to avoid clutter
                step_metrics[f"env_{i}_ongoing_return"] = self.episode_returns[i]
            
            wandb.log(step_metrics)
        
        # Track for moving average
        self.step_rewards_history.append(avg_reward)
        if len(self.step_rewards_history) > self.avg_step_reward_window:
            self.step_rewards_history.pop(0)
        
        return True
    
    def _handle_episode_completion(self, env_idx: int):
        """
        Handle completion of an episode in a specific environment.
        """
        episode_return = self.episode_returns[env_idx]
        self.completed_returns.append(episode_return)
        self.episodes_finished += 1
        
        # Calculate episode length (approximate)
        episode_length = self.total_timesteps // max(1, self.episodes_finished)
        self.episode_lengths.append(episode_length)
        
        # Log episode-level metrics
        if self.log_episode_returns:
            episode_metrics = {
                "episode": self.episodes_finished,
                "episode_return": episode_return,
                "avg_episode_return": np.mean(self.completed_returns[-100:]),  # Last 100 episodes
                "episode_length": episode_length,
                "env_idx": env_idx,
                "timestep": self.total_timesteps,
            }
            wandb.log(episode_metrics)
        
        logger.debug(
            f"Episode {self.episodes_finished} finished in env {env_idx} "
            f"with return {episode_return:.2f}"
        )
        
        # Check if we should run evaluation
        if (self.do_evaluation and 
            self.episodes_finished - self.last_eval_episode >= self.eval_frequency):
            self._run_evaluation()
            self.last_eval_episode = self.episodes_finished
        
        # Reset this environment's return tracker
        self.episode_returns[env_idx] = 0.0

    def _run_evaluation(self):
        """
        Run evaluation on the separate evaluation environment.
        """
        if not self.agent or not self.eval_env_manager:
            return
        
        logger.info(f"Running evaluation after episode {self.episodes_finished}")
        
        # Temporarily disable training for evaluation
        was_training = self.agent.training_enabled
        self.agent.disable_training()
        
        eval_returns = []
        eval_lengths = []
        
        try:
            for ep_idx in range(self.eval_episodes):
                obs = self.eval_env_manager.reset()
                done = False
                episode_return = 0.0
                episode_length = 0
                
                while not done:
                    # Get action from agent (no exploration during eval)
                    action = self.agent.choose_action(obs, deterministic=True)
                    
                    # Step the environment
                    obs, reward, done, info = self.eval_env_manager.step(action)
                    
                    episode_return += reward
                    episode_length += 1
                
                eval_returns.append(episode_return)
                eval_lengths.append(episode_length)
                
                logger.debug(
                    f"Eval episode {ep_idx + 1}/{self.eval_episodes}: "
                    f"return={episode_return:.2f}, length={episode_length}"
                )
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            eval_returns = []
        
        finally:
            # Restore training state
            if was_training:
                self.agent.enable_training()
        
        # Log evaluation results if we have any
        if eval_returns:
            mean_return = np.mean(eval_returns)
            std_return = np.std(eval_returns)
            mean_length = np.mean(eval_lengths)
            
            # Store for history
            self.eval_returns_history.append(mean_return)
            self.eval_episodes_history.append(self.episodes_finished)
            
            # Log to wandb
            eval_metrics = {
                f"{self.eval_prefix}episode": self.episodes_finished,
                f"{self.eval_prefix}mean_return": mean_return,
                f"{self.eval_prefix}std_return": std_return,
                f"{self.eval_prefix}min_return": np.min(eval_returns),
                f"{self.eval_prefix}max_return": np.max(eval_returns),
                f"{self.eval_prefix}mean_length": mean_length,
                f"{self.eval_prefix}timestep": self.total_timesteps,
            }
            
            # Add individual episode returns
            for i, ret in enumerate(eval_returns[:5]):  # Limit to first 5
                eval_metrics[f"{self.eval_prefix}ep_{i}_return"] = ret
            
            wandb.log(eval_metrics)
            
            logger.info(
                f"Evaluation complete: {self.eval_episodes} episodes, "
                f"mean return = {mean_return:.2f} ± {std_return:.2f}"
            )

    def on_learn(self, learning_info: Dict[str, Any]):
        """
        Log learning metrics (loss, entropy, etc.) from agent updates.
        """
        super().on_learn(learning_info)
        
        if learning_info:
            # Add timestep to learning info for proper alignment
            learning_info_with_step = learning_info.copy()
            learning_info_with_step["timestep"] = self.total_timesteps
            wandb.log(learning_info_with_step)

    def on_rollout_start(self):
        super().on_rollout_start()

    def on_rollout_end(self):
        super().on_rollout_end()

    def on_episode_end(self):
        super().on_episode_end()
        # Note: Episode completion is handled in on_step for parallel envs

    def _log_plots(self):
        """
        Log plot files to WandB.
        """
        if not self.plot_dir or not self.plot_dir.exists():
            return

        artifact = wandb.Artifact(f"{self.run_name}_plots", type="plots")
        for ext in ("*.png", "*.svg", "*.pdf"):
            for file in self.plot_dir.glob(ext):
                # Log as image to WandB dashboard
                wandb.log({f"plots/{file.stem}": wandb.Image(str(file))})
                # Add to artifact for download
                artifact.add_file(str(file))
                logger.debug(f"Logged plot: {file.name}")
        
        if artifact.files:
            self.run.log_artifact(artifact)

    def on_training_end(self):
        """
        Final logging and cleanup.
        """
        super().on_training_end()
        
        # Run final evaluation if enabled
        if self.do_evaluation and self.episodes_finished > self.last_eval_episode:
            logger.info("Running final evaluation...")
            self._run_evaluation()
        
        # Log final statistics
        if self.completed_returns:
            final_stats = {
                "final/avg_episode_return": np.mean(self.completed_returns),
                "final/std_episode_return": np.std(self.completed_returns),
                "final/min_episode_return": np.min(self.completed_returns),
                "final/max_episode_return": np.max(self.completed_returns),
                "final/total_episodes": len(self.completed_returns),
                "final/total_timesteps": self.total_timesteps,
                "final/avg_step_reward": np.mean(self.step_rewards_history) if self.step_rewards_history else 0,
            }
            
            # Add evaluation statistics if available
            if self.eval_returns_history:
                final_stats["final/eval_best_return"] = np.max(self.eval_returns_history)
                final_stats["final/eval_avg_return"] = np.mean(self.eval_returns_history)
                final_stats["final/eval_std_return"] = np.std(self.eval_returns_history)
            
            wandb.log(final_stats)
        
        # Log any plots
        self._log_plots()
        
        # Create and log summary CSVs
        self._log_summary_csv()
        self._log_eval_summary_csv()
        
        # Finish WandB run
        wandb.finish()
        logger.info("WandbStepCallback training ended and WandB run closed.")

    def _log_summary_csv(self):
        """
        Create and log a summary CSV of training episode returns.
        """
        if not self.completed_returns:
            return
        
        import pandas as pd
        import tempfile
        
        # Create summary DataFrame
        summary_data = {
            "episode": list(range(1, len(self.completed_returns) + 1)),
            "return": self.completed_returns,
            "length": self.episode_lengths[:len(self.completed_returns)],
        }
        
        df = pd.DataFrame(summary_data)
        
        # Save to temporary file and log as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            artifact = wandb.Artifact(f"{self.run_name}_training_summary", type="metrics")
            artifact.add_file(f.name, name="training_episode_summary.csv")
            self.run.log_artifact(artifact)
        
        logger.debug("Logged training episode summary CSV to WandB artifacts")

    def _log_eval_summary_csv(self):
        """
        Create and log a summary CSV of evaluation results.
        """
        if not self.eval_returns_history:
            return
        
        import pandas as pd
        import tempfile
        
        # Create evaluation summary DataFrame
        eval_data = {
            "training_episode": self.eval_episodes_history,
            "eval_mean_return": self.eval_returns_history,
        }
        
        df = pd.DataFrame(eval_data)
        
        # Save to temporary file and log as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            artifact = wandb.Artifact(f"{self.run_name}_eval_summary", type="metrics")
            artifact.add_file(f.name, name="evaluation_summary.csv")
            self.run.log_artifact(artifact)
        
        logger.debug("Logged evaluation summary CSV to WandB artifacts")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for external use.
        """
        stats = {
            "total_timesteps": self.total_timesteps,
            "episodes_finished": self.episodes_finished,
            "avg_episode_return": np.mean(self.completed_returns) if self.completed_returns else 0,
            "ongoing_returns": self.episode_returns.tolist() if self.episode_returns is not None else [],
        }
        
        if self.eval_returns_history:
            stats["eval_returns_history"] = self.eval_returns_history
            stats["eval_best_return"] = np.max(self.eval_returns_history)
            stats["eval_episodes_history"] = self.eval_episodes_history
        
        return stats
