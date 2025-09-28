import os
from functools import partial
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from src.agents.agentfactory import AgentFactory
from src.hyperparam_tuning.helperfunctions import eval_rl_agent, train_rl_agent, create_rl_agent
from src.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from src.metrics.metrictrackerNEW import MetricsTracker
from src.simulation.callbacks.metrictrackercallback import MetricTrackerCallback
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
            # tracker.import_metrics(metrics_path)
            tracker.load_results_from_dir(metrics_path, cfg.agent.agent_type, cfg.environment)
        tracker.set_save_path(cfg.results_save_path)

        # logger.add_metrics_tracker(tracker)

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

                callbacks=[]
                cb = MetricTrackerCallback(metric_tracker=tracker, run_id=run_idx)
                if cfg.wandb.use_wandb:
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
                if early_stop_cb:
                    callbacks.append(early_stop_cb)

                sim = SimulatorRL(exp_id, cfg.agent.agent_type, env_manager, agent,
                                  num_episodes=cfg.num_episodes,
                                  callbacks=callbacks)
                sim.train()
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
        env = gym.make(cfg.environment)
        bo = BayesianOptimizer(
            search_space=OmegaConf.to_container(cfg.mode.hpo.search_space, resolve=True),
            model_factory=partial(create_rl_agent, env=env),
            train_fn=partial(train_rl_agent, env=env, agent_id=cfg.agent.agent_type,
                             callbacks=[early_stop_cb] if early_stop_cb else None,
                             # early_stop_check=cfg.early_stopping.early_stop_check if cfg.early_stopping.enable else None,
                             # early_stop_window=cfg.early_stopping.early_stop_window if cfg.early_stopping.enable else None,
                             # early_stop_threshold=cfg.early_stopping.early_stop_threshold if cfg.early_stopping.enable else None),
            ),
            eval_fn=partial(eval_rl_agent, env=env, agent_id=cfg.agent.agent_type),
            objective_name=cfg.mode.hpo.objective_name,
            minimize=cfg.mode.hpo.minimize,
            wandb_logger=logger
        )
        best = bo.optimize(n_trials=cfg.mode.hpo.n_trials)
        print("Best hyperparameters found:", best)

        logger.finish()


if __name__ == '__main__':
    main()
