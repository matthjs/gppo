"""
This file is for estimating time and space complexity
"""
import time
import torch
import numpy as np
import torch
import numpy as np
import time
from src.agents.agent import Agent


def measure_action_latency(agent: Agent, obs: np.ndarray, n_iter: int = 1000) -> float:
    # Warm-up GPU to avoid first call overhead
    for _ in range(10):
        _ = agent.choose_action(obs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = agent.choose_action(obs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    end = time.perf_counter()
    avg_latency = (end - start) / n_iter
    print(f"Average action latency: {avg_latency * 1000:.3f} ms")
    return avg_latency

def measure_training_memory_and_time(agent):
    """
    Measure training latency and peak GPU memory usage.
    """
    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    info = agent.learn()  # Runs GPPO update
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    latency = end - start
    print(f"Training step latency: {latency:.3f} s")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
    else:
        print("No GPU detected — consider using `memory_profiler` for CPU.")

    return latency, peak_mem


def fill_rollout_buffer(agent, n_envs: int = 1):
    """
    Fill the agent's rollout buffer with fake (random) data.

    :param agent: Instance of GPPOAgent or PPOAgent with a rollout_buffer.
    :param n_envs: Number of parallel environments used.
    """
    obs_dim = agent.state_dimensions[-1]
    act_dim = agent.action_dimensions[-1]
    n_steps = agent.n_steps

    for _ in range(n_steps):
        obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
        actions = np.random.randn(n_envs, act_dim).astype(np.float32)
        rewards = np.random.randn(n_envs).astype(np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        log_probs = np.random.randn(n_envs).astype(np.float32)
        values = np.random.randn(n_envs).astype(np.float32)

        agent.rollout_buffer.push(
            obs,
            actions,
            rewards,
            dones,
            log_prob=log_probs,
            value=values
        )

    # Simulate the "last" state and done flag
    # This is a bit akward
    agent.next_state = np.random.randn(n_envs, obs_dim).astype(np.float32)
    agent.last_done = torch.zeros(n_envs, dtype=torch.float32, device=agent.device)



if __name__ == "__main__":
    pass
