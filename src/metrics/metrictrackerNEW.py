import os
import threading
import traceback
from collections import defaultdict
from typing import Union, SupportsFloat, Any, Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def interquartile_mean(values: List[float]) -> float:
    """
    Compute the interquartile mean (IQM), i.e., the mean of the middle 50% of the data.
    """
    if not values:
        return float('nan')
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    lower = int(np.floor(0.25 * n))
    upper = int(np.ceil(0.75 * n))
    trimmed = sorted_vals[lower:upper]
    return float(np.mean(trimmed))


class MetricsTracker:
    """
    Thread-safe object for recording various metrics across multiple runs.
    Tracks the interquartile mean (IQM) and stratified bootstrap-based confidence intervals of each metric,
    aggregated over multiple runs.

    Simplified per-run storage layout:
      - `per_run_values` is a mapping from run_id -> list of rows, where each row is a dict
        {"metric", "agent", "episode", "value"}. This ordering is convenient for
        saving/loading CSVs per run and mirrors a single-file-per-run export.

    The aggregated structure `_metrics_history` remains the primary data source used by
    `get_iqm_ci`, `plot_metric`, and `plot_metric_for_each_agent` (so all plotting works
    exactly as before).
    """

    def __init__(self,
                 n_bootstrap: int = 100,
                 ci_alpha: float = 0.05):
        """
        :param n_bootstrap how many bootstrap replicates are used to estimate (1-ci_alpha)*100%
        confidence interval.
        :param ci_alpha
        """
        self._lock = threading.Lock()
        # metric_name -> agent_id -> episode_idx -> List[float]
        # (aggregated values across runs; each appended value is considered one run's result)
        self._metrics_history: Dict[str, Dict[str, Dict[int, List[float]]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Simplified per-run store: run_id -> list of rows {metric, agent, episode, value}
        self._per_run_values: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha
        self.register_metric("loss")
        self.register_metric("return")

        self.save_path = "./"

    def set_save_path(self, path: str) -> None:
        self.save_path = path

    def register_metric(self, metric_name: str) -> None:
        """
        Register a new metric to be tracked.
        """
        with self._lock:
            _ = self._metrics_history[metric_name]

    def record_metric(self,
                      metric_name: str,
                      agent_id: str,
                      episode_idx: int,
                      value: Union[float, int, SupportsFloat],
                      run_id: Optional[str] = None) -> None:
        """
        Record a value for a specific metric, agent, and episode.

        - Aggregated value is appended to `_metrics_history` to support IQM/plotting.
        - A row is also appended to `per_run_values[run_id]` (or to a placeholder run id)
          to allow easy CSV export per run.
        """
        with self._lock:
            ep_i = int(episode_idx)
            v = float(value)
            # aggregated history (list-of-values for IQM computation)
            self._metrics_history[metric_name][agent_id][ep_i].append(v)
            # simple per-run row storage
            rid = run_id or "__no_run_id__"
            row = {
                "metric": metric_name,
                "agent": agent_id,
                "episode": int(ep_i),
                "value": float(v),
            }
            self._per_run_values[rid].append(row)

    # ------------------- CSV save/load per run & directory helpers -------------------
    def save_run_csv(self,
                    root_dir: str,
                    algo: str,
                    environment: str,
                    run_id: str,
                    filename: str = "metrics.csv") -> None:
        """
        Save a single run (identified by run_id) to:
            <root_dir>/<environment>/<algo>/<run_id>/<filename>
        """
        out_dir = os.path.join(root_dir, environment, algo, run_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)

        with self._lock:
            all_rows = list(self._per_run_values.get(run_id, []))
            # filter rows to only include those that match the algorithm name
            rows = [dict(r, **{"run_id": run_id}) for r in all_rows if r.get("agent") == algo]

        df = pd.DataFrame(rows, columns=["metric", "agent", "episode", "value", "run_id"]) if rows else pd.DataFrame(columns=["metric", "agent", "episode", "value", "run_id"])
        df.to_csv(out_path, index=False)

    def save_all_runs(self, root_dir: str, algo: str, environment: str, filename: str = "metrics.csv") -> None:
        """
        Save all runs present in `per_run_values` under the structured folders:
            <root_dir>/<environment>/<algo>/<run_id>/metrics.csv
        """
        with self._lock:
            run_ids = [rid for rid in self._per_run_values.keys() if rid != "__no_run_id__"]

        for rid in run_ids:
            self.save_run_csv(root_dir, algo, environment, rid, filename=filename)

    def load_results_from_dir(self,
                            root_dir: str,
                            algo: str,
                            environment: str,
                            runs: Optional[Iterable[str]] = None,
                            filename: str = "metrics.csv",
                            clear_first: bool = False) -> None:
        """
        Load CSVs from the directory structure:
            <root_dir>/<environment>/<algo>/<run_id>/<filename>
        """
        base = os.path.join(root_dir, environment, algo)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Directory not found: {base}")

        if clear_first:
            with self._lock:
                self._metrics_history.clear()
                self._per_run_values.clear()

        # discover runs
        if runs is None:
            runs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

        for run_id in runs:
            run_path = os.path.join(base, run_id, filename)
            if not os.path.isfile(run_path):
                continue
            try:
                df = pd.read_csv(run_path)
            except Exception:
                traceback.print_exc()
                continue

            # normalize column names
            cols_map = {c.lower(): c for c in df.columns}
            if "metric" not in cols_map and "name" in cols_map:
                cols_map["metric"] = cols_map["name"]
            if "value" not in cols_map and "val" in cols_map:
                cols_map["value"] = cols_map["val"]

            required = ["metric", "agent", "episode", "value"]
            if not all(k in cols_map for k in required):
                continue

            for _, row in df.iterrows():
                try:
                    metric = str(row[cols_map["metric"]])
                    agent = str(row[cols_map["agent"]])
                    episode = int(row[cols_map["episode"]])
                    value = float(row[cols_map["value"]])
                    rid = str(row[cols_map["run_id"]]) if "run_id" in cols_map else run_id
                    self.record_metric(metric, agent, episode, value, run_id=rid)
                except Exception:
                    continue

    def save_env_aggregated_plots(self,
                                root_dir: str,
                                env: str,
                                metrics: Optional[List[str]] = None,
                                output_dir: Optional[str] = None,
                                smoothing_window: int = 50) -> None:
        """
        Produce aggregated plots (across all algorithms) for the provided metrics (or all metrics if None)
        and save them into:
            <output_dir or root_dir>/<env>/plots/

        This expects that you have already loaded runs for this environment into this MetricsTracker.
        """
        out_base = output_dir or os.path.join(root_dir, env, "plots")
        os.makedirs(out_base, exist_ok=True)

        with self._lock:
            all_metrics = list(self._metrics_history.keys())
        if metrics is None:
            metrics = all_metrics

        prev_save = self.save_path
        self.save_path = out_base

        try:
            for m in metrics:
                try:
                    self.plot_metric(m, f"{m}.png", smoothing_window=smoothing_window)
                except Exception:
                    traceback.print_exc()
                    continue
        finally:
            self.save_path = prev_save



    # ------------------- IQM / plotting / utilities -------------------
    def get_iqm_ci(self,
                   metric_name: str,
                   agent_id: str,
                   episode_idx: int) -> Tuple[float, float, float]:
        """
        Compute the IQM and stratified bootstrap-based CI for a given metric, agent, and episode.
        Returns (iqm, lower_ci, upper_ci).
        """
        values = self._metrics_history.get(metric_name, {}).get(agent_id, {}).get(episode_idx, [])
        if not values:
            return float('nan'), float('nan'), float('nan')

        # Point estimate
        iqm = interquartile_mean(values)
        n = len(values)

        # Stratified bootstrap: sample run indices (0..n-1) with replacement
        bs_iqms = []
        for _ in range(self.n_bootstrap):
            idxs = np.random.randint(0, n, size=n)
            sample = [values[i] for i in idxs]
            bs_iqms.append(interquartile_mean(sample))

        lower = float(np.percentile(bs_iqms, 100 * (self.ci_alpha / 2)))
        upper = float(np.percentile(bs_iqms, 100 * (1 - self.ci_alpha / 2)))
        return iqm, lower, upper

    def plot_all_metrics(self,
                         num_episodes: int = None,
                         x_axis_label: str = "Episodes",
                         smoothing_window: int = 50) -> None:
        """
        Runs plot_metric for all stored metrics.
        """
        for metric_name in list(self._metrics_history.keys()):
            self.plot_metric(metric_name,
                             metric_name,
                             num_episodes,
                             x_axis_label,
                             y_axis_label=f"{metric_name} IQM with 95% CI",
                             title=" ",
                             smoothing_window=smoothing_window)

    def plot_metric(self,
                    metric_name: str,
                    file_name: str,
                    num_episodes: int = None,
                    x_axis_label: str = "Episodes",
                    y_axis_label: str = "Metric Value",
                    title: str = None,
                    smoothing_window: int = 50,
                    ribbon: bool = True,
                    err_bars: bool = False,
                    agent_order: List[str] = None) -> None:
        """
        Plot the IQM and stratified bootstrap-based CI over episodes for a specific metric across multiple runs,
        using a CI ribbon plus sparse error-bar caps for clarity.
        """
        with self._lock:
            if metric_name not in self._metrics_history:
                raise ValueError(f"Metric '{metric_name}' not found")

            sns.set(style="darkgrid")
            fig, ax = plt.subplots(figsize=(10, 8))

            # Convert agents to list for index tracking
            agent_items = list(self._metrics_history[metric_name].items())
            # Sort agents if order is specified
            if agent_order:
                order_map = {agent: i for i, agent in enumerate(agent_order)}
                agent_items.sort(
                    key=lambda x: order_map.get(x[0], float('inf'))  # Unknown agents last
                )

            num_agents = len(agent_items)
            max_offset = 80  # Maximum horizontal shift (adjust based on x-axis scale)

            for idx, (agent_id, episodes) in enumerate(agent_items):
                # Sort and optionally truncate
                sorted_eps = sorted(episodes.items())
                if num_episodes:
                    sorted_eps = sorted_eps[:num_episodes]
                eps = [ep for ep, _ in sorted_eps]

                # Compute horizontal offset for this agent
                if num_agents > 1:
                    offset = ((idx / (num_agents - 1)) * 2 - 1) * max_offset
                else:
                    offset = 0

                # Gather IQM and CI bounds
                iqm_vals, lower_vals, upper_vals = [], [], []
                for ep in eps:
                    iqm, low, high = self.get_iqm_ci(metric_name, agent_id, ep)
                    iqm_vals.append(iqm)
                    lower_vals.append(low)
                    upper_vals.append(high)

                # Apply smoothing if requested
                smooth_iqm_vals = iqm_vals
                lower_vals_smooth = lower_vals
                upper_vals_smooth = upper_vals
                if smoothing_window > 1:
                    smooth_iqm_vals = pd.Series(iqm_vals).rolling(window=smoothing_window,
                                                                  min_periods=1).mean().tolist()
                    lower_vals_smooth = pd.Series(lower_vals).rolling(window=smoothing_window,
                                                                      min_periods=1).mean().tolist()
                    upper_vals_smooth = pd.Series(upper_vals).rolling(window=smoothing_window,
                                                                      min_periods=1).mean().tolist()

                # Create shortened label by removing 'Agent' suffix if present
                display_label = agent_id.replace("Agent", "") if agent_id.endswith("Agent") else agent_id
                if display_label == "RANDOM":
                    display_label = "Random"

                # Plot the main IQM line
                line, = ax.plot(eps, smooth_iqm_vals, label=display_label, linewidth=2.5, alpha=1, zorder=1)
                color = line.get_color()

                # 1) CI ribbon
                if ribbon:
                    ax.fill_between(eps, lower_vals_smooth, upper_vals_smooth, color=color, alpha=0.2)

                # 2) Sparse error-bar caps at 5 evenly spaced points
                if err_bars:
                    num_marks = 5
                    if len(eps) >= num_marks:
                        indices = np.linspace(0, len(eps) - 1, num_marks, dtype=int)
                    else:
                        indices = range(len(eps))
                    for i in indices:
                        y_err_lower = iqm_vals[i] - lower_vals[i]
                        y_err_upper = upper_vals[i] - iqm_vals[i]
                        # inside your sparse-errorbar loop:
                        # Plot black outline first with thicker lines and caps
                        ax.errorbar(
                            x=eps[i] + offset,
                            y=iqm_vals[i],
                            yerr=[[y_err_lower], [y_err_upper]],
                            fmt='none',
                            color='black',
                            alpha=1,
                            linewidth=5,  # Thicker line for outline
                            capsize=6,  # Slightly larger cap size
                            capthick=3,  # Thicker cap line width
                            zorder=2  # Draw behind main errorbar
                        )
                        # Then plot the colored errorbar on top
                        ax.errorbar(
                            x=eps[i] + offset,
                            y=iqm_vals[i],
                            yerr=[[y_err_lower], [y_err_upper]],
                            fmt='none',
                            color=color,
                            alpha=1,
                            linewidth=4,
                            capsize=5,
                            capthick=2,
                            zorder=3
                        )

            # Labels, title, legend, and styling
            ax.set_title(
                title or f'{metric_name.capitalize()} IQM with {int((1 - self.ci_alpha) * 100)}% CI',
                fontsize=18
            )
            ax.set_xlabel(x_axis_label, fontsize=16)
            ax.set_ylabel(y_axis_label, fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, file_name))
            plt.savefig(os.path.join(self.save_path, file_name) + ".svg")
            plt.close()

    def clear_metrics(self) -> None:
        """
        Clear the recorded metrics for all agents and all metrics.
        """
        with self._lock:
            self._metrics_history.clear()
            self._per_run_values.clear()

    @property
    def metrics_history(self) -> dict:
        """
        Get the entire history of recorded metrics.
        """
        with self._lock:
            return self._metrics_history

    def per_run_values(self) -> dict:
        """
        Expose per-run stored values (useful for debugging / inspecting before saving).
        """
        with self._lock:
            return dict(self._per_run_values)


if __name__ == "__main__":
    experiment_id = "exp_001"
    env = "CartPole-v1"
    algos = ["PPO", "RandomAgent"]
    out_root = "results"

    # ------------------- Step 1: Initialize MetricsTracker -------------------
    mt = MetricsTracker(n_bootstrap=200)

    # ------------------- Step 2: Add initial runs (run_001 and run_002) -------------------
    rng = np.random.RandomState(0)
    initial_runs = [0, 1]
    for run in initial_runs:
        for ep in range(1, 201):
            mt.record_metric("return", "PPO", ep, 20 + 0.1 * ep + rng.randn() * 2.0 + (0.5 if run == "run_002" else 0.0), run_id=run)
            mt.record_metric("return", "RandomAgent", ep, 20 + rng.randn() * 5.0, run_id=run)

    # ------------------- Step 3: Save CSVs and aggregated plots -------------------
    for algo in algos:
        mt.save_all_runs(os.path.join(out_root, experiment_id), algo, env)
    mt.save_env_aggregated_plots(os.path.join(out_root, experiment_id), env)

    print("Initial runs saved under:", os.path.join(out_root, experiment_id, env, "plots"))

    # ------------------- Step 4: Reload existing results -------------------
    mt_reload = MetricsTracker(n_bootstrap=200)
    for algo in algos:
        mt_reload.load_results_from_dir(os.path.join(out_root, experiment_id), algo, env)

    print("Reloaded metrics. Existing runs per algorithm:")
    for algo in algos:
        for run_id, rows in mt_reload.per_run_values().items():
            if rows and any(r['agent'] == algo for r in rows):
                print(f"  {algo}, run {run_id}, {len(rows)} rows")

    # ------------------- Step 5: Add new runs (run_003 and run_004) -------------------
    rng_new = np.random.RandomState(42)
    new_runs = [3, 4]
    for run in new_runs:
        for ep in range(1, 201):
            mt_reload.record_metric("return", "PPO", ep, 20 + 0.1 * ep + rng_new.randn() * 2.0, run_id=run)
            mt_reload.record_metric("return", "RandomAgent", ep, 20 + rng_new.randn() * 5.0, run_id=run)

    # ------------------- Step 6: Save updated CSVs and aggregated plots -------------------
    for algo in algos:
        mt_reload.save_all_runs(os.path.join(out_root, experiment_id), algo, env)
    mt_reload.save_env_aggregated_plots(os.path.join(out_root, experiment_id), env)

    print("New runs added, CSVs and aggregated plots updated under:",
          os.path.join(out_root, experiment_id, env, "plots"))
