import logging
import time
from typing import Optional
from src.simulation.callbacks.abstractcallback import AbstractCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TimeBudgetCallback(AbstractCallback):
    """
    Callback that optionally stops training after N timesteps and tracks timing metrics.
    
    Tracks and reports:
    - Average inference time per step (action selection)
    - Average training time per update (when rollout buffer is full)
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
            logger.info(f"Starting timing inference and training steps...")
        self.step_start_time = time.time()
    
    def on_step(self, action, reward, next_obs, done) -> bool:
        """
        Track step timing and check if budget exceeded.
        """
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
        
        # Progress update
        if self.max_timesteps and self.verbose and self.num_steps % 1000 == 0:
            logger.info(f"Progress: {self.num_steps}/{self.max_timesteps} timesteps")
        
        # Start timing next step
        self.step_start_time = time.time()
        
        return True  # Continue training
    
    def on_update_start(self) -> None:
        """Mark the start of a training update (when rollout buffer is full)."""
        self.update_start_time = time.time()
    
    def on_update_end(self) -> None:
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
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            print(f"\nInference (per step):")
            print(f"  Average time: {avg_inference*1000:.3f} ms")
            print(f"  Total time: {sum(self.inference_times):.2f} s")
        
        if self.training_times:
            avg_training = sum(self.training_times) / len(self.training_times)
            print(f"\nTraining (per update when buffer full):")
            print(f"  Average time: {avg_training:.3f} s")
            print(f"  Total time: {sum(self.training_times):.2f} s")
            print(f"  Number of updates: {len(self.training_times)}")
        
        print("="*60 + "\n")
    
    def get_metrics(self) -> dict:
        """
        Get timing metrics as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing average times.
        """
        metrics = {
            'total_timesteps': self.num_steps,
            'num_updates': len(self.training_times),
        }
        
        if self.inference_times:
            metrics['avg_inference_time_ms'] = (
                sum(self.inference_times) / len(self.inference_times) * 1000
            )
            metrics['total_inference_time_s'] = sum(self.inference_times)
            
        if self.training_times:
            metrics['avg_training_time_s'] = (
                sum(self.training_times) / len(self.training_times)
            )
            metrics['total_training_time_s'] = sum(self.training_times)
        
        return metrics
