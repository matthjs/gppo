from typing import Optional
import torch
import wandb

from src.agents.agent import Agent
from src.agents.gpreinforceagent import GPReinforceAgent
from src.util.environment import CatchEnv
from src.agents.agentfactory import AgentFactory
from src.metrics.metrictracker import MetricsTracker


def agent_env_loop(
        agent: Agent,
        num_episodes: int,
        wandb_logger,
        # tracker: Optional[MetricsTracker] = None,
        learning: bool = True,
        env=None,
        verbose: bool = False,
        save_model: bool = False
) -> float:
    """
    Run the environment-agent interaction loop.

    :param agent: The RL agent.
    :param num_episodes: Number of training episodes.
    :param tracker: Optional metrics tracker.
    :param learning: Whether to train the agent (True) or just evaluate (False).
    :param env: Optional environment. Defaults to CatchEnv.
    :param verbose: If True, prints per-episode rewards.
    :return: Average return over all episodes.
    """
    print("Running agent on environment...")
    if not env:
        env = CatchEnv()

    total_return = 0.0

    start_episode = getattr(agent, "current_episode", 0)

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
                    if not isinstance(agent, GPReinforceAgent):
                        learning_info = agent.learn()
                        if learning_info and wandb_logger:
                            wandb_logger.log(learning_info, agent_id=type(agent).__name__,
                                             episode=episode)
                        # if tracker and learning_info:
                        #     for key, value in learning_info.items():
                        #        tracker.record_metric(key, agent_id=type(agent).__name__,
                        #                              episode_idx=episode, value=value)

                episode_return += reward
                obs = next_obs

                if terminated or truncated:
                    # This is akward but for now acceptable
                    # if isinstance(agent, GPReinforceAgent):
                    #     learning_info = agent.learn()
                    #     if tracker and learning_info:
                    #         for key, value in learning_info.items():
                    #            tracker.record_metric(key, agent_id=type(agent).__name__,
                    #                                  episode_idx=episode, value=value)

                    if verbose:
                        print(f"episode {episode} | reward: {episode_return}")

                    if wandb_logger:
                        wandb_logger.log({"return": episode_return}, agent_id=type(agent).__name__,
                                         episode=episode)
                    # tracker.record_metric("return", agent_id=type(agent).__name__,
                    #                          episode_idx=episode, value=episode_return)
                    total_return += episode_return
                    break
    except KeyboardInterrupt:
        print("Training interrupted by user!")
        if save_model:
            print("TODO, implement saving the model.")

    env.close()
    return total_return / num_episodes


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
