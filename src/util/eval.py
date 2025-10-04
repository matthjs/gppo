"""
Evaluation Experiment. E.g., show how the entropy behaves under distribution shift.
"""
import os
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from src.agents.agentfactory import AgentFactory
from src.metrics.metrictrackerNEW import MetricsTracker
from src.simulation.callbacks.metrictrackercallback import MetricTrackerCallback
from src.simulation.callbacks.wandbcallback import WandbCallback
from src.simulation.envmanager import EnvManager


def evaluation_exp(cfg: DictConfig) -> None:
    """
    For mujoco environments the physics can be manipulated
    by unwrapping the gym env. from gym.make
    base_env = env.unwrapped
    and accessing:
        >>> base_env.model.opt
        <MjOption
        ...
        gravity: array([ 0.  ,  0.  , -9.81])
        ...
        magnetic: array([ 0. , -0.5,  0. ])
        ...
        wind: array([0., 0., 0.])
        ...
        viscosity: 0.0
    e.g. change to gravity to that of the moon instead of earth:
    base_env.model.opt.gravity[:] = np.array([0.0, 0.0, -1.62])

    TODO: add method to simulatorRL actually
    """
    # Tolerat ethe code duplication for now
    # Convention. Assume training info directory exists then created save directory
    # with same name but with eval appended to it
    exp_id = cfg.exp_id
    metrics_path = os.path.join(cfg.results_save_path, exp_id)
    tracker = MetricsTracker(n_bootstrap=cfg.num_bootstrap_samples)
    save_path = os.path.join(cfg.results_save_path, "eval")
    tracker.set_save_path(save_path)

    for run_idx in range(cfg.num_runs):
        # NOTE to self: Some code duplication w.r.t. main
        # Log each run
        info_env = gym.make(cfg.environment)
        agent = AgentFactory.create_cfagent(
            cfg.agent.agent_type,
            info_env,
            cfg.n_envs,
            OmegaConf.to_container(cfg.agent.agent_params, resolve=True))
        env_manager = EnvManager(
            env_id=cfg.environment,
            env_fn=lambda: gym.make(cfg.environment),
            n_envs=cfg.n_envs,
            use_subproc=True,
            norm_obs=cfg.normalize_obs
        )

        callbacks = []
        cb = MetricTrackerCallback(metric_tracker=tracker, run_id=run_idx)
        if cfg.wandb.use_wandb:
            # Also put metrics in Wandb?
            wandb_callback = WandbCallback(
                metric_callback=cb,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=dict(cfg),
                run_name=cfg.exp_id + f"{cfg.agent.agent_type}_run{run_idx}",
                plot_dir=cfg.results_save_path  # Log plots from the results save path
            )
            callbacks.append(wandb_callback)
        else:
            callbacks.append(cb)
