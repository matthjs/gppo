"""
1. Perfect calibration: Coverage curve follows the diagonal
   - 50% intervals contain 50% of actual returns
   - 90% intervals contain 90% of actual returns

2. Overconfident: Curve BELOW diagonal
   - Intervals are too narrow
   - Predicted uncertainty underestimates actual error
   - Example: 90% intervals only capture 70% of returns
   
3. Underconfident: Curve ABOVE diagonal  
   - Intervals are too wide
   - Predicted uncertainty overestimates actual error
   - Example: 50% intervals capture 70% of returns

4. Calibration Error (CE_reg):
   - Average absolute deviation from perfect calibration
   - Lower is better (0 = perfect)
   - Typical good values: < 0.1

5. Remember: This measures calibration relative to MC returns,
   not absolute truth! Still useful for:
   - Comparing different models
   - Tuning uncertainty estimates
   - Risk-aware decision making
"""
"""
Value Function Calibration Callback - Fixed to use GAE returns from rollout buffer

Key changes:
1. Accesses rollout_buffer from agent to get actual GAE-computed returns
2. Collects data AFTER GAE computation but BEFORE buffer clearing
3. Properly matches value predictions to their training targets
"""
import logging
import os
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ValueCalibrationCallback(AbstractCallback):
    """
    Callback that collects value predictions and their GAE-computed return targets
    from the agent's rollout buffer to assess calibration.
    
    Collects data after GAE computation (in agent.learn()) to ensure we're using
    the exact same targets the value function was trained on.
    """
    
    def __init__(
        self,
        run_id: Union[str, int],
        save_path: str = "./",
        min_samples_for_calibration: int = 1000,
        alphas: Optional[np.ndarray] = None,
        compute_on_end: bool = True,
        collect_every_n_updates: int = 1,
    ):
        """
        Args:
            run_id: Identifier for this run
            save_path: Directory to save calibration plots
            min_samples_for_calibration: Minimum samples before computing calibration
            alphas: Confidence levels to test (default: 0.05 to 0.95)
            compute_on_end: Whether to compute and plot calibration at training end
            collect_every_n_updates: Collect data every N policy updates (1 = every update)
        """
        super().__init__()
        self.run_id = str(run_id) if isinstance(run_id, int) else run_id
        self.save_path = save_path
        self.min_samples = min_samples_for_calibration
        self.alphas = alphas if alphas is not None else np.linspace(0.05, 0.95, 19)
        self.compute_on_end = compute_on_end
        self.collect_every_n_updates = collect_every_n_updates
        
        # Data storage - collect from rollout buffer
        self.collected_states: List[np.ndarray] = []
        self.collected_value_means: List[float] = []
        self.collected_value_stds: List[float] = []
        self.collected_returns: List[float] = []
        
        # Metadata
        self.agent = None
        self.n_env = None
        self.agent_id = None
        self.experiment_id = None
        self.env_id = None
        self.device = None
        self.update_count = 0
        
    def init_callback(self, data: SimulatorRLData) -> None:
        """Initialize callback with simulator data."""
        super().init_callback(data)
        self.agent = data.agent
        self.n_env = data.n_env
        self.agent_id = data.agent_id
        self.experiment_id = data.experiment_id
        self.env_id = data.env_id
        
        # Get device from agent
        self.device = getattr(self.agent, 'device', torch.device('cpu'))
        
        logger.info(f"[{self.experiment_id}] ValueCalibrationCallback initialized for run {self.run_id}")
    
    def _get_value_distribution(self, obs_tensor: torch.Tensor) -> tuple:
        """
        Extract value mean and std from agent's value function.
        
        Args:
            obs_tensor: State tensor [batch_size, state_dim]
            
        Returns:
            value_means: [batch_size] predicted value means
            value_stds: [batch_size] predicted value standard deviations
        """
        self.agent.policy.eval()
        
        with torch.no_grad():
            # Get value distribution from policy
            _, value_dist = self.agent.policy(obs_tensor)
            
            # Extract mean and std (handles DSPP/DGP structure)
            if hasattr(self.agent.policy, 'quad_weights'):
                # Deep Sigma Point Process (DSPP) - aggregate across quadrature points
                quad_weights = self.agent.policy.quad_weights.unsqueeze(-1).exp()
                value_mean = (quad_weights * value_dist.mean).sum(0)
                
                # Total variance: within + between variance
                within_var = (quad_weights * value_dist.variance).sum(0)
                between_var = (quad_weights * (value_dist.mean - value_mean.unsqueeze(0))**2).sum(0)
                value_std = torch.sqrt(within_var + between_var)
            else:
                # Standard distribution (e.g., heteroscedastic network)
                value_mean = value_dist.mean
                value_std = torch.sqrt(value_dist.variance)
            
            return value_mean, value_std
    
    def on_step(self, action, reward, next_obs, done) -> bool:
        """
        Called at each environment step. 
        
        We don't collect data here - we collect after GAE computation in the learn() call.
        """
        super().on_step(action, reward, next_obs, done)
        
        # Check if buffer is full and agent will call learn()
        # if len(self.agent.rollout_buffer) >= self.agent.batch_size:
        #     self.update_count += 1
            
        # Only collect data every N updates
        if self.update_count % self.collect_every_n_updates == 0:
            self._collect_from_buffer()
        
        return True
    
    def _collect_from_buffer(self):
        """
        Collect value predictions and GAE returns from the rollout buffer.
        
        This should be called AFTER agent computes GAE but BEFORE buffer.clear().
        We hook into this via on_step when buffer is full.
        """
        buffer = self.agent.rollout_buffer
        
        if buffer.pos == 0:
            return
        
        try:
            # Get states from buffer
            states = buffer.obs[:buffer.pos]  # [buffer_size, state_dim]
            
            # Get the GAE-computed returns (the actual training targets!)
            gae_returns = buffer.returns[:buffer.pos]  # [buffer_size]
            
            # Get current value predictions for these states
            value_means, value_stds = self._get_value_distribution(states)
            
            # Store data
            self.collected_states.extend(states.cpu().numpy())
            self.collected_value_means.extend(value_means.cpu().numpy())
            self.collected_value_stds.extend(value_stds.cpu().numpy())
            self.collected_returns.extend(gae_returns.cpu().numpy())
            
            logger.debug(f"Collected {buffer.pos} samples from rollout buffer")
            
        except Exception as e:
            logger.warning(f"Failed to collect from buffer: {e}")
    
    def _compute_coverage_curve(self) -> tuple:
        """
        Compute calibration curve from collected data.
        
        Returns:
            alphas: Nominal confidence levels
            coverages: Empirical coverages
            ce_reg: Calibration error
        """
        returns = np.array(self.collected_returns)
        value_means = np.array(self.collected_value_means)
        value_stds = np.array(self.collected_value_stds)
        
        coverages = []
        
        for alpha in self.alphas:
            # Two-sided z-score
            z = norm.ppf((1 + alpha) / 2.0)
            
            # Prediction intervals
            lower = value_means - z * value_stds
            upper = value_means + z * value_stds
            
            # Empirical coverage
            covered = np.mean((returns >= lower) & (returns <= upper))
            coverages.append(covered)
        
        coverages = np.array(coverages)
        ce_reg = np.mean(np.abs(self.alphas - coverages))
        
        return self.alphas, coverages, ce_reg
    
    def _plot_calibration(self, alphas: np.ndarray, coverages: np.ndarray, 
                          ce_reg: float, save_path: str):
        """
        Create and save calibration plot.
        """
        plt.figure(figsize=(7, 7))
        plt.plot(alphas, coverages, 'o-', linewidth=2, markersize=6, 
                 label='Empirical coverage', color='#2E86AB')
        plt.plot([0, 1], [0, 1], '--', linewidth=2, 
                 label='Perfect calibration', color='#A23B72')
        
        # Shade the region between curve and diagonal
        plt.fill_between(alphas, alphas, coverages, alpha=0.2, 
                         color='#2E86AB' if coverages.mean() > alphas.mean() else '#E63946')
        
        plt.xlabel('Nominal Confidence Level (α)', fontsize=12)
        plt.ylabel('Empirical Coverage', fontsize=12)
        plt.title(f'Value Function Calibration (Run {self.run_id})\nCE = {ce_reg:.4f}', 
                  fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved calibration plot to {save_path}")
    
    def compute_and_plot_calibration(self) -> Optional[Dict]:
        """
        Compute calibration metrics and create plot.
        
        Returns:
            Dictionary with calibration results, or None if insufficient data
        """
        n_samples = len(self.collected_returns)
        
        if n_samples == 0:
            logger.warning("No samples collected - cannot compute calibration")
            return None
        
        if n_samples < self.min_samples:
            logger.warning(f"Only {n_samples} samples collected (min: {self.min_samples}) - calibration may be unreliable")
        
        logger.info(f"Computing calibration from {n_samples} state-value pairs...")
        
        # Compute calibration
        alphas, coverages, ce_reg = self._compute_coverage_curve()
        
        # Statistics
        value_means = np.array(self.collected_value_means)
        value_stds = np.array(self.collected_value_stds)
        returns = np.array(self.collected_returns)
        
        # Prediction error statistics
        errors = returns - value_means
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        
        logger.info(f"Calibration Error (CE_reg): {ce_reg:.4f}")
        logger.info(f"Mean predicted value: {value_means.mean():.3f} ± {value_means.std():.3f}")
        logger.info(f"Mean GAE return: {returns.mean():.3f} ± {returns.std():.3f}")
        logger.info(f"Mean predicted std: {value_stds.mean():.3f}")
        logger.info(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        
        # Create plot
        os.makedirs(self.save_path, exist_ok=True)
        plot_path = os.path.join(
            self.save_path, 
            f"{self.agent_id}_calibration_run{self.run_id}.png"
        )
        self._plot_calibration(alphas, coverages, ce_reg, plot_path)
        
        return {
            'ce_reg': float(ce_reg),
            'n_samples': n_samples,
            'mean_predicted_value': float(value_means.mean()),
            'std_predicted_value': float(value_means.std()),
            'mean_gae_return': float(returns.mean()),
            'std_gae_return': float(returns.std()),
            'mean_predicted_std': float(value_stds.mean()),
            'mae': float(mae),
            'rmse': float(rmse),
            'alphas': alphas.tolist(),
            'coverages': coverages.tolist()
        }
    
    def on_training_end(self) -> None:
        """
        Called at the end of training/evaluation.
        
        Computes and saves calibration plot if enabled.
        """
        super().on_training_end()
        
        if not self.compute_on_end:
            return
        
        # Compute calibration
        results = self.compute_and_plot_calibration()
        
        if results:
            # Save numerical results
            json_path = os.path.join(
                self.save_path,
                f"{self.agent_id}_calibration_run{self.run_id}.json"
            )
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved calibration results to {json_path}")
