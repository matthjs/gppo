import os
from functools import partial
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from src.agents.agentfactory import AgentFactory
from src.hyperparam_tuning.helperfunctions import eval_rl_agent, train_rl_agent, create_rl_agent
from src.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from src.metrics.metrictrackerNEW import MetricsTracker
from src.simulation.callbacks.agentcheckpointcallback import AgentCheckpointCallback
from src.simulation.callbacks.metrictrackercallback import MetricTrackerCallback
from src.simulation.callbacks.vecnormalizecallback import VecNormalizeCallback
from src.simulation.callbacks.wandbcallback import WandbCallback
from src.simulation.envmanager import EnvManager
from src.simulation.simulator_rl import SimulatorRL
from src.util.wandblogger import WandbLogger
from src.simulation.callbacks.earlystopcallback import EarlyStopCallback


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

    if cfg.mode.name == 'train':
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

        if cfg.mode.train:
            for run_idx in range(cfg.num_runs):
                # Log each run
                info_env = gym.make(cfg.environment)
                agent = AgentFactory.create_agent(
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

                # ---- Callbacks ----
                callbacks=[]
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
                    ))
                # Save and/or load VecNormalize stats?
                if cfg.normalize_obs:
                    callbacks.append(
                        VecNormalizeCallback(
                            save_base=cfg.results_save_path,
                            run_id=run_idx,
                            load_on_start=False,    # For now just always set it to false
                            save_on_end=True   # For now, always do this
                    ))

                sim = SimulatorRL(exp_id, cfg.agent.agent_type, env_manager, agent,
                                  num_episodes=cfg.num_episodes,
                                  callbacks=callbacks)
                sim.train()
        else:
            print("No training, running metric tracker plotting method...")
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
            train_fn=partial(train_rl_agent, env_id=cfg.environment, agent_id=cfg.agent.agent_type,
                             normalize_obs=cfg.normalize_obs,
                             callbacks=[early_stop_cb] if early_stop_cb else None),   # We only are interested in the early_stop callback here.
            eval_fn=partial(eval_rl_agent, env_id=cfg.environment, agent_id=cfg.agent.agent_type),
            objective_name=cfg.mode.hpo.objective_name,
            minimize=cfg.mode.hpo.minimize,
            wandb_logger=logger,
            save_path=os.path.join(cfg.results_save_path, cfg.exp_id, cfg.agent.agent_type, cfg.environment)
        )
        best = bo.optimize(n_trials=cfg.mode.hpo.n_trials)
        print("Best hyperparameters found:", best)
        logger.finish()


if __name__ == '__main__':
    main()
