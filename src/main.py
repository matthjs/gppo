import os
from functools import partial
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from src.agents.agentfactory import AgentFactory
from src.hyperparam_tuning.helperfunctions import eval_rl_agent, train_rl_agent, create_rl_agent
from src.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from src.metrics.metrictracker import MetricsTracker
from src.simulation.callbacks.metrictrackercallback import MetricTrackerCallback
from src.simulation.envmanager import EnvManager
from src.simulation.simulator_rl import SimulatorRL
from src.util.interaction import agent_env_loop
from src.util.wandblogger import WandbLogger


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Entrypoint of program. See hydra configs.
    """
    print(OmegaConf.to_yaml(cfg))

    logger = WandbLogger(
        enable=cfg.wandb.use_wandb,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=dict(cfg),
        name=cfg.mode.name
    )

    if cfg.results_save_path:
        os.makedirs(cfg.results_save_path, exist_ok=True)

    if cfg.mode.name == 'train':
        metrics_path = os.path.join(cfg.results_save_path, "metrics.json")
        tracker = MetricsTracker(n_bootstrap=cfg.num_bootstrap_samples)
        if cfg.mode.import_metrics:
            tracker.import_metrics(metrics_path)
        tracker.set_save_path(cfg.results_save_path)

        logger.add_metrics_tracker(tracker)

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
                    env_fn=lambda: gym.make(cfg.environment),
                    n_envs=cfg.n_envs,
                    use_subproc=True,
                    norm_obs=cfg.normalize_obs
                )

                sim = SimulatorRL(env_manager, agent,
                                  num_episodes=cfg.num_episodes,
                                  callbacks=[MetricTrackerCallback(tracker)])
                sim.train()
                
                """
                agent_env_loop(agent, cfg.num_episodes, logger, run_idx, learning=True, env=env, verbose=True,
                               save_model=cfg.mode.save_model,
                               normalize_obs=cfg.normalize_obs,
                               normalize_action=cfg.normalize_act,
                               save_path=cfg.results_save_path,
                               early_stop_check=cfg.early_stopping.early_stop_check if cfg.early_stopping.enable else None,
                               early_stop_window=cfg.early_stopping.early_stop_window if cfg.early_stopping.enable else None,
                               early_stop_threshold=cfg.early_stopping.early_stop_threshold if cfg.early_stopping.enable else None)
                logger.log_metric_tracker_state(cfg.num_episodes, cfg.mode.export_metrics)
                logger.finish()
                """
        # if not cfg.wandb.use_wandb:
        #    tracker.plot_all_metrics(num_episodes=cfg.num_episodes)
        # After plotting, get final metrics
        final_stats = tracker.get_final_metrics("return")
        for agent, stats in final_stats.items():
            print(f"{agent} return:")
            print(f"  Final Episode: {stats['episode']}")
            print(f"  IQM: {stats['iqm']:.2f} (95% CI: {stats['lower_ci']:.2f}-{stats['upper_ci']:.2f})\n")


    elif cfg.mode.name == 'hpo' or cfg.mode.name == 'hpo_gppo':
        logger.start()
        env = gym.make(cfg.environment)
        bo = BayesianOptimizer(
            search_space=OmegaConf.to_container(cfg.mode.hpo.search_space, resolve=True),
            model_factory=partial(create_rl_agent, env=env),
            train_fn=partial(train_rl_agent, env=env,
                             early_stop_check=cfg.early_stopping.early_stop_check if cfg.early_stopping.enable else None,
                             early_stop_window=cfg.early_stopping.early_stop_window if cfg.early_stopping.enable else None,
                             early_stop_threshold=cfg.early_stopping.early_stop_threshold if cfg.early_stopping.enable else None),
            eval_fn=partial(eval_rl_agent, env=env),
            objective_name=cfg.mode.hpo.objective_name,
            minimize=cfg.mode.hpo.minimize,
            wandb_logger=logger
        )
        best = bo.optimize(n_trials=cfg.mode.hpo.n_trials)
        print("Best hyperparameters found:", best)

        logger.finish()


if __name__ == '__main__':
    main()
