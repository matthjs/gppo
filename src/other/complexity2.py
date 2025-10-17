import time
import torch
import numpy as np
import psutil
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO as SB3_PPO
from src.agents.gppoagent import GPPOAgent
from src.agents.sbadapter import StableBaselinesAdapter


@dataclass
class ScalingMetrics:
    """Store scaling benchmark results."""
    config_name: str
    agent_name: str
    n_steps: int  # Rollout buffer size
    model_size: int
    num_parameters: int
    
    # Inference metrics (per action)
    inference_time_ms: float
    inference_time_std_ms: float
    inference_memory_mb: float
    
    # Training metrics (per update)
    training_time_s: float
    training_time_std_s: float
    training_memory_mb: float
    
    # Throughput
    inference_throughput: float  # actions/sec
    training_throughput: float   # samples/sec


class MemoryTracker:
    """Track memory usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = 0
        
    def start(self):
        """Record baseline memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def get_current_usage(self) -> float:
        """Get current memory usage above baseline in MB."""
        current = self.process.memory_info().rss / 1024 / 1024
        return current - self.baseline_memory
    
    def get_cuda_memory(self) -> float:
        """Get CUDA memory if available."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0


class AgentBenchmark:
    """
    Comprehensive benchmarking framework for comparing RL agents.
    Measures time and space complexity across different scales.
    """
    
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        n_envs: int = 1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize benchmark.
        
        Args:
            env_name: Gymnasium environment name
            n_envs: Number of parallel environments
            device: PyTorch device (auto-detected if None)
        """
        self.env_name = env_name
        self.n_envs = n_envs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create environment to inspect dimensions
        self.env = gym.make(env_name)
        self.state_dim = self._get_state_dim()
        self.action_dim = self._get_action_dim()
        
        print(f"Initialized AgentBenchmark:")
        print(f"  Environment: {env_name}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Device: {self.device}")
        print(f"  n_envs: {n_envs}")
    
    def _get_state_dim(self) -> Tuple:
        """Extract state dimension from environment."""
        return self.env.observation_space.shape
    
    def _get_action_dim(self) -> Tuple:
        """Extract action dimension from environment."""
        if hasattr(self.env.action_space, 'shape') and len(self.env.action_space.shape) > 0:
            # Continuous action space
            return self.env.action_space.shape
        else:
            # Discrete action space
            return (self.env.action_space.n,)
    
    def create_gppo_agent(
        self,
        model_scale: int = 1,
        n_steps: int = 256
    ) -> GPPOAgent:
        """
        Create GPPO agent with specified configuration.
        Override this method to customize GPPO initialization.
        
        Args:
            model_scale: Multiplier for layer dimensions
            n_steps: Rollout buffer size per environment
            
        Returns:
            Configured GPPO agent
        """
        base_dim = 6 * model_scale
        batch_size = max(64, (n_steps * self.n_envs) // 4)
        
        hidden_layers_config = [
            {'output_dims': base_dim, 'mean_type': 'linear'}
        ]
        
        policy_hidden_config = [
            {'output_dims': base_dim, 'mean_type': 'constant'}
        ]
        
        value_hidden_config = [
            {'output_dims': None, 'mean_type': 'constant'}
        ]
        
        agent = GPPOAgent(
            state_dimensions=self.state_dim,
            action_dimensions=self.action_dim,
            n_steps=n_steps,
            n_envs=self.n_envs,
            batch_size=batch_size,
            learning_rate=0.005751,
            n_epochs=9,
            gamma=0.999,
            gae_lambda=0.9,
            clip_range=0.350164,
            ent_coef=0.001684,
            vf_coef=0.835695,
            max_grad_norm=0.664338,
            num_inducing_points=128,
            num_quad_sites=8,
            beta=0.5,
            sample_vf=True,
            hidden_layers_config=hidden_layers_config,
            policy_hidden_config=policy_hidden_config,
            value_hidden_config=value_hidden_config,
            device=self.device
        )
        
        return agent
    
    def create_sb3_agent(
        self,
        model_scale: int = 1,
        n_steps: int = 256
    ) -> StableBaselinesAdapter:
        """
        Create SB3 PPO agent with specified configuration.
        Override this method to customize SB3 initialization.
        
        Args:
            model_scale: Multiplier for network dimensions
            n_steps: Rollout buffer size per environment
            
        Returns:
            Configured SB3 PPO agent wrapped in adapter
        """
        batch_size = max(64, (n_steps * self.n_envs) // 4)
        
        net_arch = [dict(
            pi=[64 * model_scale, 64 * model_scale], 
            vf=[64 * model_scale, 64 * model_scale]
        )]
        
        sb3_model = SB3_PPO(
            "MlpPolicy",
            self.env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=0.005751,
            n_epochs=9,
            gamma=0.999,
            gae_lambda=0.9,
            clip_range=0.350164,
            ent_coef=0.001684,
            vf_coef=0.835695,
            max_grad_norm=0.664338,
            policy_kwargs={"net_arch": net_arch},
            verbose=0
        )
        
        return StableBaselinesAdapter(sb3_model)
    
    @staticmethod
    def count_parameters(model) -> int:
        """Count total trainable parameters in a model."""
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0
    
    def benchmark_inference(
        self,
        agent,
        num_steps: int = 500,
        warmup_steps: int = 50
    ) -> Tuple[float, float, float, float]:
        """
        Benchmark inference performance.
        
        Args:
            agent: Agent to benchmark
            num_steps: Number of inference steps
            warmup_steps: Number of warmup steps
            
        Returns:
            (mean_time_ms, std_time_ms, memory_mb, throughput_actions_per_sec)
        """
        memory_tracker = MemoryTracker()
        times = []
        
        # Warmup
        obs, _ = self.env.reset()
        for _ in range(warmup_steps):
            _ = agent.choose_action(obs)
        
        # Benchmark
        memory_tracker.start()
        obs, _ = self.env.reset()
        
        for _ in range(num_steps):
            start = time.perf_counter()
            action = agent.choose_action(obs)
            end = time.perf_counter()
            
            times.append(end - start)
            obs, _, done, truncated, _ = self.env.step(action)
            
            if done or truncated:
                obs, _ = self.env.reset()
        
        memory_usage = memory_tracker.get_current_usage()
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0
        
        return mean_time * 1000, std_time * 1000, memory_usage, throughput
    
    def benchmark_training(
        self,
        agent,
        n_steps: int,
        num_updates: int = 5
    ) -> Tuple[float, float, float, float]:
        """
        Benchmark training performance.
        
        Args:
            agent: Agent to benchmark
            n_steps: Rollout buffer size
            num_updates: Number of training updates
            
        Returns:
            (mean_time_s, std_time_s, memory_mb, throughput_samples_per_sec)
        """
        memory_tracker = MemoryTracker()
        times = []
        
        memory_tracker.start()
        
        for _ in range(num_updates):
            obs, _ = self.env.reset()
            start = time.perf_counter()
            
            if isinstance(agent, GPPOAgent):
                # Collect rollout
                for step in range(n_steps):
                    action = agent.choose_action(obs)
                    new_obs, reward, done, truncated, _ = self.env.step(action)
                    agent.store_transition(obs, action, reward, new_obs, done or truncated)
                    obs = new_obs
                    
                    if done or truncated:
                        obs, _ = self.env.reset()
                
                # Train on collected rollout
                _ = agent.learn()
                
            elif isinstance(agent, StableBaselinesAdapter):
                sb_model = agent.stable_baselines_unwrapped()
                # SB3 internally collects n_steps and trains
                sb_model.learn(total_timesteps=n_steps * self.n_envs, reset_num_timesteps=False)
            
            end = time.perf_counter()
            times.append(end - start)
        
        memory_usage = memory_tracker.get_current_usage()
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = (n_steps * self.n_envs) / mean_time if mean_time > 0 else 0
        
        return mean_time, std_time, memory_usage, throughput
    
    def run_scaling_benchmark(
        self,
        n_steps_list: List[int] = [128, 256, 512, 1024],
        model_scales: List[int] = [1, 2, 4],
        inference_steps: int = 500,
        training_updates: int = 5,
        benchmark_gppo: bool = True,
        benchmark_sb3: bool = True
    ) -> pd.DataFrame:
        """
        Run complete scaling analysis.
        
        Args:
            n_steps_list: List of rollout buffer sizes to test
            model_scales: List of model scale multipliers
            inference_steps: Number of inference steps per benchmark
            training_updates: Number of training updates per benchmark
            benchmark_gppo: Whether to benchmark GPPO
            benchmark_sb3: Whether to benchmark SB3
            
        Returns:
            DataFrame with all benchmark results
        """
        results = []
        
        total_configs = len(n_steps_list) * len(model_scales)
        if benchmark_gppo:
            total_configs *= 1
        if benchmark_sb3:
            total_configs *= 1
        total_configs = total_configs * (int(benchmark_gppo) + int(benchmark_sb3))
        
        config_idx = 0
        
        print("\n" + "=" * 80)
        print(f"SCALING BENCHMARK: {total_configs} configurations")
        print("=" * 80)
        
        for model_scale in model_scales:
            for n_steps in n_steps_list:
                rollout_buffer_size = n_steps * self.n_envs
                
                # Benchmark GPPO
                if benchmark_gppo:
                    config_idx += 1
                    print(f"\n[{config_idx}/{total_configs}] GPPO: model_scale={model_scale}, n_steps={n_steps} (buffer={rollout_buffer_size})")
                    
                    try:
                        agent = self.create_gppo_agent(model_scale, n_steps)
                        
                        params = self.count_parameters(agent.policy)
                        print(f"  Parameters: {params:,}")
                        print(f"  Batch size: {agent.batch_size}")
                        
                        print("  Benchmarking inference...")
                        inf_time, inf_std, inf_mem, inf_throughput = \
                            self.benchmark_inference(agent, inference_steps)
                        
                        print("  Benchmarking training...")
                        train_time, train_std, train_mem, train_throughput = \
                            self.benchmark_training(agent, n_steps, training_updates)
                        
                        results.append(ScalingMetrics(
                            config_name=f"scale{model_scale}_nsteps{n_steps}",
                            agent_name="GPPO",
                            n_steps=rollout_buffer_size,
                            model_size=model_scale,
                            num_parameters=params,
                            inference_time_ms=inf_time,
                            inference_time_std_ms=inf_std,
                            inference_memory_mb=inf_mem,
                            training_time_s=train_time,
                            training_time_std_s=train_std,
                            training_memory_mb=train_mem,
                            inference_throughput=inf_throughput,
                            training_throughput=train_throughput
                        ))
                        
                        print(f"  ✓ Inference: {inf_time:.2f}ms, Training: {train_time:.2f}s")
                        print(f"  ✓ Throughput: {train_throughput:.2f} samples/sec")
                        
                        del agent
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        print(f"  ✗ ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Benchmark SB3
                if benchmark_sb3:
                    config_idx += 1
                    print(f"\n[{config_idx}/{total_configs}] SB3: model_scale={model_scale}, n_steps={n_steps} (buffer={rollout_buffer_size})")
                    
                    try:
                        agent = self.create_sb3_agent(model_scale, n_steps)
                        
                        params = self.count_parameters(agent.stable_baselines_unwrapped().policy)
                        batch_size = agent.stable_baselines_unwrapped().batch_size
                        print(f"  Parameters: {params:,}")
                        print(f"  Batch size: {batch_size}")
                        
                        print("  Benchmarking inference...")
                        inf_time, inf_std, inf_mem, inf_throughput = \
                            self.benchmark_inference(agent, inference_steps)
                        
                        print("  Benchmarking training...")
                        train_time, train_std, train_mem, train_throughput = \
                            self.benchmark_training(agent, n_steps, training_updates)
                        
                        results.append(ScalingMetrics(
                            config_name=f"scale{model_scale}_nsteps{n_steps}",
                            agent_name="SB3_PPO",
                            n_steps=rollout_buffer_size,
                            model_size=model_scale,
                            num_parameters=params,
                            inference_time_ms=inf_time,
                            inference_time_std_ms=inf_std,
                            inference_memory_mb=inf_mem,
                            training_time_s=train_time,
                            training_time_std_s=train_std,
                            training_memory_mb=train_mem,
                            inference_throughput=inf_throughput,
                            training_throughput=train_throughput
                        ))
                        
                        print(f"  ✓ Inference: {inf_time:.2f}ms, Training: {train_time:.2f}s")
                        print(f"  ✓ Throughput: {train_throughput:.2f} samples/sec")
                        
                        del agent
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        print(f"  ✗ ERROR: {e}")
                        import traceback
                        traceback.print_exc()
        
        df = pd.DataFrame([asdict(r) for r in results])
        return df
    
    def plot_results(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create comprehensive scaling plots.
        
        Args:
            df: Results DataFrame
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'GPPO vs SB3 PPO: Scaling Analysis\n{self.env_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Inference Time vs Rollout Buffer Size
        ax = axes[0, 0]
        for agent in df['agent_name'].unique():
            data = df[df['agent_name'] == agent].groupby('n_steps')['inference_time_ms'].mean()
            ax.plot(data.index, data.values, marker='o', label=agent, linewidth=2, markersize=8)
        ax.set_xlabel('Rollout Buffer Size (n_steps)', fontsize=11)
        ax.set_ylabel('Inference Time (ms)', fontsize=11)
        ax.set_title('Inference Time Scaling', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        # 2. Training Time vs Rollout Buffer Size
        ax = axes[0, 1]
        for agent in df['agent_name'].unique():
            data = df[df['agent_name'] == agent].groupby('n_steps')['training_time_s'].mean()
            ax.plot(data.index, data.values, marker='o', label=agent, linewidth=2, markersize=8)
        ax.set_xlabel('Rollout Buffer Size (n_steps)', fontsize=11)
        ax.set_ylabel('Training Time (s)', fontsize=11)
        ax.set_title('Training Time Scaling', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        # 3. Memory vs Rollout Buffer Size
        ax = axes[0, 2]
        for agent in df['agent_name'].unique():
            data = df[df['agent_name'] == agent].groupby('n_steps')['training_memory_mb'].mean()
            ax.plot(data.index, data.values, marker='o', label=agent, linewidth=2, markersize=8)
        ax.set_xlabel('Rollout Buffer Size (n_steps)', fontsize=11)
        ax.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax.set_title('Training Memory Scaling', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        # 4. Inference Time vs Model Size
        ax = axes[1, 0]
        for agent in df['agent_name'].unique():
            data = df[df['agent_name'] == agent].groupby('model_size')['inference_time_ms'].mean()
            ax.plot(data.index, data.values, marker='s', label=agent, linewidth=2, markersize=8)
        ax.set_xlabel('Model Scale Factor', fontsize=11)
        ax.set_ylabel('Inference Time (ms)', fontsize=11)
        ax.set_title('Inference Time vs Model Size', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Training Time vs Model Size
        ax = axes[1, 1]
        for agent in df['agent_name'].unique():
            data = df[df['agent_name'] == agent].groupby('model_size')['training_time_s'].mean()
            ax.plot(data.index, data.values, marker='s', label=agent, linewidth=2, markersize=8)
        ax.set_xlabel('Model Scale Factor', fontsize=11)
        ax.set_ylabel('Training Time (s)', fontsize=11)
        ax.set_title('Training Time vs Model Size', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Throughput Comparison
        ax = axes[1, 2]
        gppo_data = df[df['agent_name'] == 'GPPO'].groupby('n_steps')['training_throughput'].mean()
        sb3_data = df[df['agent_name'] == 'SB3_PPO'].groupby('n_steps')['training_throughput'].mean()
        
        x = np.arange(len(gppo_data))
        width = 0.35
        ax.bar(x - width/2, gppo_data.values, width, label='GPPO', alpha=0.8)
        ax.bar(x + width/2, sb3_data.values, width, label='SB3_PPO', alpha=0.8)
        ax.set_xlabel('Rollout Buffer Size (n_steps)', fontsize=11)
        ax.set_ylabel('Throughput (samples/sec)', fontsize=11)
        ax.set_title('Training Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(gppo_data.index)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlots saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics and comparative analysis."""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        for agent in df['agent_name'].unique():
            agent_df = df[df['agent_name'] == agent]
            print(f"\n{agent}:")
            print(f"  Avg Inference Time: {agent_df['inference_time_ms'].mean():.2f} ± {agent_df['inference_time_ms'].std():.2f} ms")
            print(f"  Avg Training Time: {agent_df['training_time_s'].mean():.2f} ± {agent_df['training_time_s'].std():.2f} s")
            print(f"  Avg Parameters: {agent_df['num_parameters'].mean():.0f}")
            print(f"  Avg Inference Memory: {agent_df['inference_memory_mb'].mean():.2f} MB")
            print(f"  Avg Training Memory: {agent_df['training_memory_mb'].mean():.2f} MB")
            print(f"  Avg Training Throughput: {agent_df['training_throughput'].mean():.2f} samples/sec")
        
        # Comparative analysis
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS")
        print("=" * 80)
        
        gppo_df = df[df['agent_name'] == 'GPPO']
        sb3_df = df[df['agent_name'] == 'SB3_PPO']
        
        if len(gppo_df) > 0 and len(sb3_df) > 0:
            inf_ratio = gppo_df['inference_time_ms'].mean() / sb3_df['inference_time_ms'].mean()
            train_ratio = gppo_df['training_time_s'].mean() / sb3_df['training_time_s'].mean()
            param_ratio = gppo_df['num_parameters'].mean() / sb3_df['num_parameters'].mean()
            
            print(f"\nGPPO vs SB3:")
            print(f"  Inference Time Ratio: {inf_ratio:.2f}x {'(slower)' if inf_ratio > 1 else '(faster)'}")
            print(f"  Training Time Ratio: {train_ratio:.2f}x {'(slower)' if train_ratio > 1 else '(faster)'}")
            print(f"  Parameter Ratio: {param_ratio:.2f}x {'(more)' if param_ratio > 1 else '(fewer)'}")
    
    def __del__(self):
        """Clean up environment."""
        if hasattr(self, 'env'):
            self.env.close()


# Example usage
if __name__ == "__main__":
    # Create benchmark instance
    benchmark = AgentBenchmark(
        env_name="Walker2d-v5",
        n_envs=1
    )
    
    # Run scaling benchmark
    results_df = benchmark.run_scaling_benchmark(
        n_steps_list=[128, 256, 512, 1024],
        model_scales=[1, 2, 4],
        inference_steps=500,
        training_updates=5,
        benchmark_gppo=True,
        benchmark_sb3=True
    )
    
    # Save results
    results_df.to_csv("scaling_benchmark_results.csv", index=False)
    print("\nResults saved to scaling_benchmark_results.csv")
    
    # Print summary
    benchmark.print_summary(results_df)
    
    # Create plots
    benchmark.plot_results(results_df, save_path="scaling_plots.png")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
