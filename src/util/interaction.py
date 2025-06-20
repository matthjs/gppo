from collections import deque

import torch
import os
import gymnasium as gym
from typing import Optional
from src.agents.agent import Agent
from src.util.actionrescaler import ActionRescaleWrapper
from src.util.environment import CatchEnv
from src.agents.agentfactory import AgentFactory
from src.util.vecnormalize import VecNormalizeGymEnv
from src.util.wandblogger import WandbLogger


def agent_env_loop(
        agent: Agent,
        num_episodes: int,
        wandb_logger: WandbLogger,
        learning: bool = True,
        env: gym.Env = None,
        verbose: bool = False,
        save_model: bool = False,
        load_model: bool = False,
        normalize_obs: bool = True,
        normalize_action: bool = False,
        save_path: str = "./",
        early_stop_check: int = None,
        early_stop_window: int = None,
        early_stop_threshold: float = None
) -> float:
    """
    Run the environment-agent interaction loop.

    :param agent: The RL agent.
    :param num_episodes: Number of training episodes.
    :param wandb_logger: Optional wandb logger, can be computed with a MetricTracker.
    :param learning: Whether to train the agent (True) or just evaluate (False).
    :param env: Optional environment. Defaults to CatchEnv.
    :param verbose: If True, prints per-episode rewards.
    :param save_model:
    :param normalize_obs:
    :return: Average return over all episodes.
    """
    print("Running agent on environment...")
    if not env:
        env = CatchEnv()

    total_return = 0.0

    start_episode = getattr(agent, "current_episode", 0)

    if normalize_action:
        env = ActionRescaleWrapper(env, new_low=-1, new_high=1)

    if normalize_obs:
        env = VecNormalizeGymEnv(env, norm_obs=True)
        if not learning:
            print("Not training, loading normalization stats")
            env.training = False
            env.load(os.path.join(save_path, "obs_norm_stats.pkl"))

    if load_model:
        agent.load(os.path.join(save_path, "type(agent).__name__"))

    reward_buffer = deque(maxlen=early_stop_window) if early_stop_window else None

    try:
        for episode in range(start_episode, start_episode + num_episodes):
            episode_return = 0
            obs, info = env.reset()

            while True:
                action = agent.choose_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                if learning:
                    agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
                    agent.update()
                    learning_info = agent.learn()
                    if learning_info and wandb_logger:
                        wandb_logger.log(learning_info, agent_id=type(agent).__name__,
                                            episode=episode)
                episode_return += reward
                obs = next_obs

                if terminated or truncated:
                    if verbose:
                        print(f"episode {episode} | reward: {episode_return}")

                    if wandb_logger:
                        wandb_logger.log({"return": episode_return}, agent_id=type(agent).__name__,
                                         episode=episode)
                    total_return += episode_return

                    if reward_buffer is not None:
                        reward_buffer.append(episode_return)
                        if early_stop_check and (episode + 1) % early_stop_check == 0:
                            avg_recent = sum(reward_buffer) / len(reward_buffer)
                            if avg_recent < early_stop_threshold:
                                print(f"Early stopping at episode {episode + 1} — "
                                      f"Average return over last {early_stop_window} episodes: {avg_recent:.2f} "
                                      f"(threshold: {early_stop_threshold})")
                                raise StopIteration
                    break
    except StopIteration:
        print("Early stopping triggered!")
    except KeyboardInterrupt:
        print("Training interrupted by user!")
    if save_model:
        agent.save(os.path.join(save_path, "type(agent).__name__"))
    if normalize_obs:
        env.save(os.path.join(save_path, "obs_norm_stats.pkl"))

    env.close()
    return total_return / num_episodes - start_episode + 1


def create_agent_for_catch_env(agent_type: str, num_episodes: int, agent_params: Optional[dict] = None) -> Agent:
    """
    Create an agent configured to work in the CatchEnv environment.
    The epsilon-greedy schedule will be inserted into the agent.

    :param agent_type: Agent class name registered in the AgentFactory.
    :param num_episodes: Number of episodes (also used for exploration schedule).
    :param agent_params: Optional dictionary with agent parameters.
    :return: Configured Agent instance.
    """
    env = CatchEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if agent_params is None:
        agent_params = {}

    # Set default exploration policy if not provided
    if "expl_policy_name" not in agent_params:
        agent_params["expl_policy_name"] = "epsilon_greedy"
    if "expl_policy_params" not in agent_params:
        agent_params["expl_policy_params"] = {}

    # Set default values for exploration parameters if not explicitly set
    expl_params = agent_params["expl_policy_params"]
    expl_params.setdefault("epsilon_start", 1)
    expl_params.setdefault("epsilon_end", 0)
    expl_params["epsilon_decay_steps"] = num_episodes  # always override to match run length

    # Always ensure these are injected
    agent_params.update({
        "state_dimensions": env.observation_space.shape,
        "n_actions": env.action_space.n,
        "device": device,
        "expl_policy_params": expl_params,  # ensure it's the updated dict
    })

    agent = AgentFactory.create_agent(agent_type=agent_type, env=env, agent_params=agent_params)
    return agent
