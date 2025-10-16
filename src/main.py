import os
from functools import partial
from typing import Callable, Optional
import hydra
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.agents.agentfactory import AgentFactory
from src.hyperparam_tuning.helperfunctions import eval_rl_agent, train_rl_agent, create_rl_agent
from src.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from src.metrics.metrictrackerNEW import MetricsTracker
from src.simulation.callbacks.agentcheckpointcallback import AgentCheckpointCallback
from src.simulation.callbacks.complexitycallback import TimeBudgetCallback
from src.simulation.callbacks.metrictrackercallback import MetricTrackerCallback
from src.simulation.callbacks.vecnormalizecallback import VecNormalizeCallback
from src.simulation.callbacks.wandbcallback import WandbCallback
from src.simulation.envmanager import EnvManager
from src.simulation.simulator_rl import SimulatorRL
from src.util.eval import evaluation_exp
from src.util.wandblogger import WandbLogger
from src.simulation.callbacks.earlystopcallback import EarlyStopCallback
from src.simulation.callbacks.valuecalibrationcallback import ValueCalibrationCallback

def make_simulator(
    cfg: DictConfig,
    agent_id: str,
    exp_id: str,
    run_idx: int,
    tracker: MetricsTracker,
    early_stop_cb: Optional[EarlyStopCallback],
    env_fn: Callable[[], gym.Env],
    train: bool
) -> SimulatorRL:
    """
    Factory function to create a SimulatorRL instance with agent, environment manager, and callbacks.
    """
    # Log each run
    env_manager = EnvManager(
        env_id=cfg.environment,
        env_fn=env_fn,
        n_envs=cfg.n_envs,
        use_subproc=True,
        norm_obs=cfg.normalize_obs
    )
    
    info_env = gym.make(cfg.environment)
    agent = AgentFactory.create_agent(
        cfg.agent.agent_type,
        env_manager.vec_env, # info_env,
        cfg.n_envs,
        OmegaConf.to_container(cfg.agent.agent_params, resolve=True)
    )

    # ---- Callbacks ----
    callbacks = []
    # Use MetricTracker?
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
    # Put early stopping condition?
    if early_stop_cb:
        callbacks.append(early_stop_cb)
    # Save and/or load model/RL agent?
    if cfg.mode.save_model or cfg.mode.load_model:
        callbacks.append(
            AgentCheckpointCallback(
                save_base=cfg.results_save_path,
                run_id=run_idx,   # Result will be saved per run
                load_on_start=cfg.mode.load_model,
                save_on_end=cfg.mode.save_model,
            )
        )
    # Save and/or load VecNormalize stats?
    if cfg.normalize_obs:
        callbacks.append(
            VecNormalizeCallback(
                save_base=cfg.results_save_path,
                run_id=cfg.mode.load_run_id if not train else run_idx,
                load_on_start=False if train else True,
                save_on_end=True if train else False
            )
        )

    if not train:
        calibration_cb = ValueCalibrationCallback(
            run_id=run_idx,
            save_path=os.path.join(cfg.results_save_path, exp_id, "calibration"),
            # min_episodes_for_calibration=cfg.mode.eval_episodes,
            compute_on_end=True  # Compute at end of training/eval
        )
        callbacks.append(calibration_cb)
        # Estimate
        callbacks.append(
            TimeBudgetCallback()
        )

    # callbacks.append(TimeBudgetCallback()) TODO: Make this more flexible

    sim = SimulatorRL(
        exp_id,
        agent_id,
        env_manager,
        agent,
        num_episodes=cfg.num_episodes,
        callbacks=callbacks
    )

    return sim


def execute_runs(cfg: DictConfig,
                 agent_id: str,
                 exp_id: str,
                 tracker: MetricsTracker,
                 early_stop_cb,
                 env_fn,
                 train):
    """
    Train multiple runs of the agent, setting up environments, agents, callbacks, and metrics.
    """
    for run_idx in range(cfg.num_runs if train else cfg.mode.eval_runs):
        # Log each run
        sim = make_simulator(cfg, agent_id, exp_id, run_idx, tracker,
                       early_stop_cb, env_fn, train)
        if train:
            sim.train()
        else:
            sim.evaluate(cfg.mode.eval_episodes)

def make_tweaked_mujoco_env(cfg):
    env = gym.make(cfg.environment)   # render_mode="human"
    env.reset()
    base_env = env.unwrapped
    base_env.model.opt.gravity[:] = np.array([0.0, 0.0, -1.62])
    return env

@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Entrypoint of program. See hydra configs.
    """
    print(OmegaConf.to_yaml(cfg))

    if cfg.results_save_path:
        os.makedirs(cfg.results_save_path, exist_ok=True)

    early_stop_cb = None
    if cfg.early_stopping.enable:
        early_stop_cb = EarlyStopCallback(
            early_stop_check=cfg.early_stopping.early_stop_check,
            early_stop_window=cfg.early_stopping.early_stop_window,
            early_stop_threshold=cfg.early_stopping.early_stop_threshold
        )

    if cfg.mode.name == 'train' or cfg.mode.name == 'eval':
        # metrics_path = os.path.join(cfg.results_save_path, "metrics.json")
        exp_id = cfg.exp_id
        metrics_path = os.path.join(cfg.results_save_path, exp_id)
        tracker = MetricsTracker(n_bootstrap=cfg.num_bootstrap_samples)
        if cfg.mode.import_metrics:
            for agent_name in cfg.mode.agents:
                # Dynamically import from src.agents
                try:
                    tracker.load_results_from_dir(metrics_path, agent_name, cfg.environment)
                except FileNotFoundError:
                    print(f"No existing metrics found for agent {agent_name} in {metrics_path}. Skipping.")
        tracker.set_save_path(cfg.results_save_path)

        # NOTE: This is turning into spaghetti code fix this later
        if cfg.mode.name == "train":
            if cfg.mode.execute:
                execute_runs(cfg,
                             cfg.agent.agent_type,
                             exp_id,
                             tracker,
                             early_stop_cb,
                             env_fn=lambda: gym.make(cfg.environment), train=True)
            else:
                print("No training, running metric tracker plotting method...")
                tracker.save_env_aggregated_plots(metrics_path, cfg.environment)
        elif cfg.mode.name == "eval":
            if cfg.mode.execute:
                # Hardcode this for now
                save_path = os.path.join(cfg.results_save_path, "eval")
                tracker.set_save_path(save_path)
                execute_runs(cfg, cfg.agent.agent_type, exp_id, tracker, early_stop_cb, env_fn=lambda: gym.make(cfg.environment), train=False)
                # TODO: UPDATE THIS
                # save_path = os.path.join(cfg.results_save_path, "adjusted gravity")
                # tracker.set_save_path(save_path)
                # execute_runs(cfg, cfg.agent.agent_type, exp_id, tracker, early_stop_cb, env_fn=lambda: make_tweaked_mujoco_env(cfg), train=False)
            else:
                tracker.save_env_aggregated_plots(metrics_path, cfg.environment)
            # Maybe also run sim.evaluate(...)?            
    elif cfg.mode.name == 'hpo' or cfg.mode.name == 'hpo_gppo':
        # For now, use the legacy WandbLogger class.
        logger = WandbLogger(
            enable=cfg.wandb.use_wandb,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=dict(cfg),
            name=cfg.mode.name
        )

        logger.start()
        info_env = gym.make(cfg.environment)
        bo = BayesianOptimizer(
            search_space=OmegaConf.to_container(cfg.mode.hpo.search_space, resolve=True),
            model_factory=partial(create_rl_agent, env=info_env),
            train_fn=partial(train_rl_agent, exp_id=cfg.exp_id, env_id=cfg.environment, agent_id=cfg.agent.agent_type,
                             normalize_obs=cfg.normalize_obs,
                             callbacks=[early_stop_cb if early_stop_cb else None,
                                            VecNormalizeCallback(
                                                save_base=cfg.results_save_path,
                                                run_id=0,   # Only one run per trail, just name it 0
                                                load_on_start=False,    # For now just always set it to false
                                                save_on_end=True   # For now, always do this
                                        )]),
            eval_fn=partial(eval_rl_agent, callbacks=[
                                            VecNormalizeCallback(
                                                save_base=cfg.results_save_path,
                                                run_id=0,
                                                load_on_start=True,    # For now just always set it to false
                                                save_on_end=False   # For now, always do this
                                        )], exp_id=cfg.exp_id, env_id=cfg.environment, agent_id=cfg.agent.agent_type),
            objective_name=cfg.mode.hpo.objective_name,
            minimize=cfg.mode.hpo.minimize,
            wandb_logger=logger,
            save_path=os.path.join(cfg.results_save_path, cfg.exp_id, cfg.agent.agent_type, cfg.environment)
        )
        best = bo.optimize(n_trials=cfg.mode.hpo.n_trials)
        print("Best hyperparameters found:", best)
        logger.finish()
    elif cfg.mode.name == "eval":
        evaluation_exp(cfg)


if __name__ == '__main__':
    main()
