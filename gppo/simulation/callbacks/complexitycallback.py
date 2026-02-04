import logging
import time
import numpy as np
from typing import Optional
from gppo.simulation.callbacks.abstractcallback import AbstractCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TimeBudgetCallback(AbstractCallback):
    """
    Callback that optionally stops training after N timesteps and tracks timing metrics.
    
    Tracks and reports:
    - Mean and standard deviation of inference times (action selection)
    - Mean and standard deviation of training times (when rollout buffer is full)
    """

    def __init__(self, max_timesteps: Optional[int] = None, verbose: bool = True):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.verbose = verbose
        
        # Timing tracking
        self.step_start_time = None
        self.update_start_time = None
        self.inference_times = []
        self.training_times = []
        
    def on_training_start(self) -> None:
        """Initialize timing at training start."""
        if self.verbose:
            logger.info("Starting timing inference and training steps...")
        self.step_start_time = time.time()
    
    def on_step(self, action, reward, next_obs, done) -> bool:
        """Track step timing and check if budget exceeded."""
        # Record inference time (time to select action)
        if self.step_start_time is not None:
            inference_time = time.time() - self.step_start_time
            self.inference_times.append(inference_time)
        
        # Increment step counter
        self.num_steps += 1
        self.old_obs = next_obs
        
        # Check if budget exceeded
        if self.max_timesteps and self.num_steps >= self.max_timesteps:
            if self.verbose:
                logger.info(f"\nReached timestep budget: {self.num_steps}/{self.max_timesteps}")
            return False  # Stop training
        
        # Periodic progress update
        if self.max_timesteps and self.verbose and self.num_steps % 1000 == 0:
            logger.info(f"Progress: {self.num_steps}/{self.max_timesteps} timesteps")
        
        # Start timing next step
        self.step_start_time = time.time()
        
        return True  # Continue training
    
    def on_rollout_end(self) -> None:
        """Mark the start of a training update (when rollout buffer is full)."""
        self.update_start_time = time.time()
    
    def on_learn(self, learning_info=None) -> None:
        """Record training update time (gradient updates, etc.)."""
        if self.update_start_time is not None:
            training_time = time.time() - self.update_start_time
            self.training_times.append(training_time)
            self.update_start_time = None
    
    def on_training_end(self) -> None:
        """Print summary statistics."""
        if self.verbose:
            self.print_summary()
    
    def print_summary(self) -> None:
        """Print timing metrics summary."""
        print("\n" + "="*60)
        print("TIMING METRICS SUMMARY")
        print("="*60)
        
        print(f"\nTotal timesteps: {self.num_steps}")
        
        if self.inference_times:
            mean_inf = np.mean(self.inference_times)
            std_inf = np.std(self.inference_times)
            total_inf = np.sum(self.inference_times)
            print(f"\nInference (per step):")
            print(f"  Mean time: {mean_inf*1000:.3f} ms")
            print(f"  Std dev:   {std_inf*1000:.3f} ms")
            print(f"  Total time: {total_inf:.2f} s")
        
        if self.training_times:
            mean_train = np.mean(self.training_times)
            std_train = np.std(self.training_times)
            total_train = np.sum(self.training_times)
            print(f"\nTraining (per update when buffer full):")
            print(f"  Mean time: {mean_train:.3f} s")
            print(f"  Std dev:   {std_train:.3f} s")
            print(f"  Total time: {total_train:.2f} s")
            print(f"  Number of updates: {len(self.training_times)}")
        
        print("="*60 + "\n")
    
    def get_metrics(self) -> dict:
        """
        Get timing metrics as a dictionary (with mean and std values).

        :return: dict with timing statistics
        """
        metrics = {
            'total_timesteps': self.num_steps,
            'num_updates': len(self.training_times),
        }

        if self.inference_times:
            metrics.update({
                'mean_inference_time_ms': np.mean(self.inference_times) * 1000,
                'std_inference_time_ms': np.std(self.inference_times) * 1000,
                'total_inference_time_s': np.sum(self.inference_times),
            })
        
        if self.training_times:
            metrics.update({
                'mean_training_time_s': np.mean(self.training_times),
                'std_training_time_s': np.std(self.training_times),
                'total_training_time_s': np.sum(self.training_times),
            })
        
        return metrics
