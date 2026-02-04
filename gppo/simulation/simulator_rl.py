import logging
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from gppo.agents.agent import Agent
from gppo.agents.sbadapter import StableBaselinesAdapter
from gppo.simulation.callbacks.abstractcallback import AbstractCallback
from gppo.simulation.callbacks.sbcallbackadapter import SB3CallbackAdapter
from gppo.simulation.envmanager import EnvManager
from gppo.simulation.simulatorldata import SimulatorRLData
from gppo.util.wandblogger import WandbLogger
import os
from functools import partial
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from gppo.agents.agentfactory import AgentFactory
from gppo.util.wandblogger import WandbLogger
from linear_operator.utils.errors import NotPSDError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimulatorRL:
    """High-level trainer supporting SB3 agents or custom agents."""

    def __init__(
        self,
        experiment_id: str,
        agent_id: str,
        env_manager: EnvManager,
        agent: Agent,
        num_episodes: int,
        callbacks: Optional[List[AbstractCallback]] = None,
        save_model: bool = False,
        load_model: bool = False,
        device: Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        self.agent_id = agent_id
        self.env_manager = env_manager
        self.n_env = self.env_manager.n_envs
        self.agent = agent
        self.callbacks = callbacks or []
        self.save_model = save_model
        self.load_model = load_model
        self.num_episodes = num_episodes
        self.env_id = env_manager.env_id
        self.device = device

    def _call_callbacks(self, fn_name: str, *args, **kwargs):
        return_value = True
        for cb in self.callbacks:
            ret = getattr(cb, fn_name)(*args, **kwargs)
            if ret is not None:
                return_value = return_value and ret
        return return_value

    def train(self) -> Optional[Tuple[float, float]]:
        self._call_callbacks("init_callback",
                             data=SimulatorRLData(self))
        if isinstance(self.agent, StableBaselinesAdapter):
            self._train_sb3(self.num_episodes)
            return None
        else:
            return self._env_interaction(self.num_episodes)

    def evaluate(self, num_eval_episodes) -> Tuple[float, float]:
        self._call_callbacks("init_callback", data=SimulatorRLData(self))
        self.agent.disable_training()
        ret = self._env_interaction(num_eval_episodes)
        self.agent.enable_training()
        return ret

    def _train_sb3(self, num_episodes: int = 10):
        logger.info("[{self.experiment_id}] Delegating training to SB3 agent.learn")
        # For some reason internally max_episodes is actually set to max_episodes * n_envs
        max_episode_callback = StopTrainingOnMaxEpisodes(max_episodes=num_episodes // self.n_env, verbose=0)
        sb_callbacks = [max_episode_callback]
        for callback in self.callbacks:
            sb_callbacks.append(SB3CallbackAdapter(callback))
        model = self.agent.stable_baselines_unwrapped()

        model.learn(total_timesteps=4242424242424, callback=sb_callbacks)

    def _env_interaction(self, num_episodes: int) -> Tuple[float, float]:
        """
        This method assumes we are running on episodic environments.
        Returns the mean and variance of episode returns.
        """
        logger.info(f"[{self.experiment_id}] Running custom loop: total_episodes={num_episodes}")
        self._call_callbacks("on_training_start")
        n_env = self.env_manager.n_envs
        episodes_finished = 0
        episode_returns = []  # store individual episode returns
        entropy_values = []

        try:
            obs = self.env_manager.reset()
            ep_return = np.zeros(n_env)

            while episodes_finished < num_episodes:
                self._call_callbacks("on_rollout_start")
                actions = self.agent.choose_action(obs)
                next_obs, rewards, dones, infos = self.env_manager.step(actions)
                # self.env_manager.render()

                ep_return += rewards

                # Handle finished episodes
                for i in range(n_env):
                    if dones[i]:
                        episodes_finished += 1
                        episode_returns.append(ep_return[i])
                        ep_return[i] = 0.0

                if not self._call_callbacks("on_step",
                                            action=actions,
                                            reward=rewards,
                                            next_obs=next_obs,
                                            done=dones):
                    break

                self.agent.store_transition(obs, actions, rewards, next_obs, dones)
                self.agent.update()
                if self.agent.full_buffer():
                    self._call_callbacks("on_rollout_end")
                learning_info = self.agent.learn()

                if "entropy" in learning_info:
                    entropy_values.append(learning_info["entropy"])

                self._call_callbacks("on_learn", learning_info=learning_info)
                obs = next_obs

        except StopIteration:
            logger.info(f"[{self.experiment_id}] Early stopping triggered!")
        except KeyboardInterrupt:
            logger.info(f"[{self.experiment_id}] Training interrupted by user!")
        except SystemExit:
            logger.info(f"[{self.experiment_id}] SystemExit received, stopping training!")
        except Exception as e:
            logger.info(f"[{self.experiment_id}] Other error: {e}, returning zeros for mean and variance.")
            episode_returns = []

        self._call_callbacks("on_training_end")
        self.env_manager.close()

        if len(episode_returns) == 0:
            return {"mean_return": 0.0, "std_return": 0.0}

        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        ret = {"mean_return": mean_return, "std_return": std_return}
        if len(entropy_values) > 0:
            ret["mean_entropy"] = np.mean(entropy_values)
            ret["std_entropy"] = np.std(entropy_values)

        return ret
                


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
