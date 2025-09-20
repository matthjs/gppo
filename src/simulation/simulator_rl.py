import logging
from typing import Any, Dict, List, Optional
import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from src.metrics.metrictracker import MetricsTracker
from src.simulation.callbacks.sbcallbackadapter import StableBaselinesCallbackAdapter
from src.simulation.envmanager import EnvManager
from src.util.wandblogger import WandbLogger
import os
from functools import partial
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from src.agents.agentfactory import AgentFactory
from src.hyperparam_tuning.helperfunctions import eval_rl_agent, train_rl_agent, create_rl_agent
from src.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from src.metrics.metrictracker import MetricsTracker
from src.util.interaction import agent_env_loop
from src.util.wandblogger import WandbLogger


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimulatorRL:
    """High-level trainer supporting SB3 agents or custom agents."""

    def __init__(
        self,
        env_manager: EnvManager,
        agent: Any,
        callbacks: Optional[List[Any]] = None,
        save_model: bool = False,
        load_model: bool = False,
        device: Optional[str] = None,
    ):
        self.env_manager = env_manager
        self.agent = agent
        self.callbacks = callbacks or []
        self.save_model = save_model
        self.load_model = load_model
        self.device = device

    def _call_callbacks(self, fn_name: str, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, fn_name)(self, *args, **kwargs)

    def train(self, num_episodes: int) -> None:
        self._call_callbacks("init_callback")
        if isinstance(self.agent, BaseAlgorithm):
            self._train_sb3(num_episodes)
        else:
            self._env_interaction(num_episodes, training=True)

    def _train_sb3(self, num_episodes: int = 10):
        logger.info("Delegating training to SB3 agent.learn")
        max_episode_callback = StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=0)
        sb_callbacks = [max_episode_callback]
        for callback in self.callbacks:
            callback.append(StableBaselinesCallbackAdapter(callback))
        model = self.agent.stable_baselines_unwrapped()

        model.learn(total_timesteps=4242424242424, callback=sb_callbacks)

    def _env_interaction(self, num_episodes: int, training: bool = True) -> None:
        logger.info(f"Running custom training loop: total_episodes={num_episodes}")
        self._call_callbacks("on_training_start")
        n_env = self.env_manager.n_envs

        # Track ongoing returns for each environment
        episode_returns = np.zeros(n_env, dtype=np.float32)
        completed_returns: List[float] = []
        episodes_finished = 0

        try:
            obs = self.env_manager.reset()

            while episodes_finished < num_episodes:
                actions = self.agent.choose_action(obs)
                next_obs, rewards, dones, infos = self.env_manager.step(actions)

                # Update per-env accumulators
                episode_returns += rewards

                # Handle finished episodes
                for i in range(n_env):
                    if dones[i]:
                        completed_returns.append(episode_returns[i])
                        episodes_finished += 1
                        logger.info(f"Episode {episodes_finished} finished with return {episode_returns[i]:.2f}")
                        episode_returns[i] = 0.0  # reset for next episode

                if training:
                    self.agent.store_transition(obs, actions, rewards, next_obs, dones)
                    self.agent.update()
                    learning_info = self.agent.learn()

                obs = next_obs
                self._call_callbacks("on_step")

        except StopIteration:
            print("Early stopping triggered!")
        except KeyboardInterrupt:
            print("Training interrupted by user!")
        except Exception as e:
            print(f"General error: {e}")

        if self.save_model:
            print("Saving agent ... TODO")
            # self.agent.save("agent_checkpoint.pt")  # TODO: hook into your path logic

        self._call_callbacks("on_training_end")
        self.env_manager.close()

"""
if __name__ == "__main__":
    # quick main to test things
    tracker = MetricsTracker(n_bootstrap=10)
    logger = WandbLogger(
        enable=False
    )
    logger.add_metrics_tracker(tracker)
    env_manager = EnvManager(env_fn=lambda: gym.make("CartPole-v1") , n_envs=3, use_subproc=True, norm_obs=True)
    sim = SimulatorRL(env_manager)
    sim.train(num_episodes=10)
"""

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Entrypoint of program. See hydra configs.
    """
    print(OmegaConf.to_yaml(cfg))

    logger = WandbLogger(
        enable=False
    )

    for run_idx in range(1):
        logger.start()
        info_env = gym.make(cfg.environment)
        env_manager = EnvManager(env_fn=lambda: gym.make(cfg.environment), n_envs=1, use_subproc=True, norm_obs=True)
        agent = AgentFactory.create_agent(cfg.agent.agent_type, info_env, 1, OmegaConf.to_container(cfg.agent.agent_params, resolve=True))
        sim = SimulatorRL(env_manager, agent)
        sim.train(num_episodes=cfg.num_episodes)

if __name__ == '__main__':
    main()
