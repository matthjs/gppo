import os
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from src.agents.agentfactory import AgentFactory
from src.hyperparam_tuning.helperfunctions import create_rl_agent_catch, eval_rl_agent, train_rl_agent
from src.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from src.metrics.metrictracker import MetricsTracker
from src.util.interaction import create_agent_for_catch_env, agent_env_loop


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Entrypoint of program. See hydra configs.
    """
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode.name == 'train':
        metrics_path = os.path.join(cfg.results_save_path, "metrics.json")
        tracker = MetricsTracker(n_bootstrap=cfg.num_bootstrap_samples)
        if cfg.mode.import_metrics:
            tracker.import_metrics(metrics_path)
        tracker.set_save_path(cfg.results_save_path)

        if cfg.mode.train:
            for _ in range(cfg.num_runs):
                env = gym.make("InvertedPendulum-v5")
                agent = AgentFactory.create_agent(cfg.agent.agent_type, env,
                                                  OmegaConf.to_container(cfg.agent.agent_params, resolve=True))

                # agent = create_agent_for_catch_env(cfg.agent.agent_type, cfg.num_episodes,
                #                                   OmegaConf.to_container(cfg.agent.agent_params, resolve=True))
                agent_env_loop(agent, cfg.num_episodes, tracker=tracker, learning=True, env=env, verbose=True)

        tracker.plot_all_metrics(num_episodes=cfg.num_episodes)
        # After plotting, get final metrics
        final_stats = tracker.get_final_metrics("return")
        for agent, stats in final_stats.items():
            print(f"{agent} return:")
            print(f"  Final Episode: {stats['episode']}")
            print(f"  IQM: {stats['iqm']:.2f} (95% CI: {stats['lower_ci']:.2f}-{stats['upper_ci']:.2f})\n")

        if cfg.mode.export_metrics:
            tracker.export_metrics(str(metrics_path))
    elif cfg.mode.name == 'hpo':
        bo = BayesianOptimizer(
            search_space=OmegaConf.to_container(cfg.mode.hpo.search_space, resolve=True),
            model_factory=create_rl_agent_catch,
            train_fn=train_rl_agent,

            eval_fn=eval_rl_agent,
            objective_name=cfg.mode.hpo.objective_name,
            minimize=cfg.mode.hpo.minimize,
        )
        best = bo.optimize(n_trials=cfg.mode.hpo.n_trials)
        print("Best hyperparameters found:", best)


if __name__ == '__main__':
    main()
