import ast
import json
import os
import threading
from typing import Union, SupportsFloat, Any, Dict, Tuple, List
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as pe
import wandb
from wandb.apis.public import Run
from typing import Callable, Optional, List, Dict
import traceback

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
        # metric_name -> agent_id -> episode_idx -> List of recorded values per run
        self._metrics_history: Dict[str, Dict[str, Dict[int, List[float]]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
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
                      value: Union[float, int, SupportsFloat]) -> None:
        """
        Record a value for a specific metric, agent, and episode. Each value corresponds to one run's result.
        """
        with self._lock:
            self._metrics_history[metric_name][agent_id][episode_idx].append(float(value))

    def get_iqm_ci(self,
                   metric_name: str,
                   agent_id: str,
                   episode_idx: int) -> Tuple[float, float, float]:
        """
        Compute the IQM and stratified bootstrap-based CI for a given metric, agent, and episode.
        Sampling is done at the run level: we resample run indices with replacement and
        compute the IQM on each bootstrap replicate to form the CI.
        Returns (iqm, lower_ci, upper_ci).
        """
        values = self._metrics_history[metric_name][agent_id].get(episode_idx, [])
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
        for metric_name in self._metrics_history.keys():
            self.plot_metric(metric_name,
                             metric_name,
                             num_episodes,
                             x_axis_label,
                             y_axis_label=f"{metric_name} IQM with 95% CI",
                             title=" ",
                             smoothing_window=smoothing_window)

    def get_final_metrics(self, metric_name: str) -> dict:
        """Returns final episode's IQM and CI for all agents in a dictionary."""
        final_metrics = {}
        with self._lock:
            if metric_name not in self._metrics_history:
                raise ValueError(f"Metric '{metric_name}' not found")

            for agent_id, episodes in self._metrics_history[metric_name].items():
                sorted_eps = sorted(episodes.items())
                if not sorted_eps:  # No episodes recorded
                    continue

                # Get last episode data
                last_ep = sorted_eps[-1][0]
                iqm, lower, upper = self.get_iqm_ci(metric_name, agent_id, last_ep)

                # Shorten agent name for display
                display_name = agent_id.replace("Agent", "") if agent_id.endswith("Agent") else agent_id
                final_metrics[display_name] = {
                    'episode': last_ep,
                    'iqm': iqm,
                    'lower_ci': lower,
                    'upper_ci': upper
                }

        return final_metrics

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

    def plot_metric_for_each_agent(self,
                                   metric_name: str,
                                   file_name: str,
                                   num_episodes: int = None,
                                   x_axis_label: str = "Episodes",
                                   y_axis_label: str = "Metric Value",
                                   title: str = None,
                                   smoothing_window: int = 1,
                                   ribbon: bool = True,
                                   err_bars: bool = False,
                                   baseline_agent_id: str = "RandomAgent",
                                   ncols: int = 3) -> None:
        """
        Plot each agent vs. baseline in its own subplot, all in one figure,
        with properly padded global X/Y labels.
        """
        with self._lock:
            if metric_name not in self._metrics_history:
                raise ValueError(f"Metric '{metric_name}' not found")

            # Extract histories
            all_agents = dict(self._metrics_history[metric_name])
            if baseline_agent_id not in all_agents:
                raise ValueError(f"Baseline '{baseline_agent_id}' not found")
            baseline_hist = all_agents.pop(baseline_agent_id)

            # Prepare baseline data
            sorted_base = (sorted(baseline_hist.items())[:num_episodes]
                           if num_episodes else sorted(baseline_hist.items()))
            base_eps = [ep for ep, _ in sorted_base]
            base_iqm, base_low, base_high = zip(*(
                self.get_iqm_ci(metric_name, baseline_agent_id, ep)
                for ep, _ in sorted_base
            ))
            if smoothing_window > 1:
                base_iqm = pd.Series(base_iqm).rolling(window=smoothing_window, min_periods=1).mean().tolist()
                base_low = pd.Series(base_low).rolling(window=smoothing_window, min_periods=1).mean().tolist()
                base_high = pd.Series(base_high).rolling(window=smoothing_window, min_periods=1).mean().tolist()

            # Layout calculation
            agent_ids = list(all_agents.keys())
            n_agents = len(agent_ids)
            nrows = int(np.ceil(n_agents / ncols))

            sns.set(style="darkgrid")
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(5 * ncols, 4 * nrows),
                sharex=True, sharey=True
            )
            axes = axes.flatten()

            # Per-agent subplot
            for idx, agent_id in enumerate(agent_ids):
                ax = axes[idx]
                hist = all_agents[agent_id]
                sorted_agent = (sorted(hist.items())[:num_episodes]
                                if num_episodes else sorted(hist.items()))
                eps = [ep for ep, _ in sorted_agent]
                iqm_vals, low_vals, high_vals = zip(*(
                    self.get_iqm_ci(metric_name, agent_id, ep)
                    for ep, _ in sorted_agent
                ))

                if smoothing_window > 1:
                    iqm_s = pd.Series(iqm_vals).rolling(window=smoothing_window, min_periods=1).mean().tolist()
                    low_s = pd.Series(low_vals).rolling(window=smoothing_window, min_periods=1).mean().tolist()
                    high_s = pd.Series(high_vals).rolling(window=smoothing_window, min_periods=1).mean().tolist()
                else:
                    iqm_s, low_s, high_s = iqm_vals, low_vals, high_vals

                # Baseline
                if ribbon:
                    ax.fill_between(base_eps, base_low, base_high, color="gray", alpha=0.2)
                ax.plot(base_eps, base_iqm, color="gray", lw=2, label=baseline_agent_id)

                # Agent
                if ribbon:
                    ax.fill_between(eps, low_s, high_s, alpha=0.2)
                line, = ax.plot(eps, iqm_s, lw=2.5, label="Agent")
                color = line.get_color()

                if err_bars:
                    marks = min(5, len(eps))
                    for i in np.linspace(0, len(eps) - 1, marks, dtype=int):
                        err_low = iqm_vals[i] - low_vals[i]
                        err_high = high_vals[i] - iqm_vals[i]
                        # outline
                        ax.errorbar(eps[i], iqm_vals[i],
                                    yerr=[[err_low], [err_high]],
                                    fmt='none', color='black',
                                    linewidth=4, capsize=5, capthick=2, zorder=2)
                        # colored
                        ax.errorbar(eps[i], iqm_vals[i],
                                    yerr=[[err_low], [err_high]],
                                    fmt='none', color=color,
                                    linewidth=2, capsize=4, capthick=1, zorder=3)

                ax.axhline(1, color='black', linestyle='--', alpha=0.6, lw=1)
                display_label = agent_id.replace("Agent", "") if agent_id.endswith("Agent") else agent_id
                ax.set_title(display_label, fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.4)

            # Remove unused axes
            for j in range(n_agents, len(axes)):
                fig.delaxes(axes[j])

            # Super-title and global labels with padding
            fig.suptitle(
                title or f"{metric_name.capitalize()} per Agent vs. {baseline_agent_id}",
                fontsize=18, y=0.98
            )
            fig.supxlabel(x_axis_label, fontsize=16, y=0.02)

            # Adjusted ylabel positionings
            left_margin = 0.10  # Increased from 0.10 to create more space

            fig.supylabel(y_axis_label, fontsize=16, x=left_margin - 0.04, ha='right')

            # Single legend
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right', fontsize=12)

            # Adjust margins to avoid overlap
            fig.subplots_adjust(
                left=left_margin,  # More space for y-label
                bottom=0.10,  # Space for x-label
                top=0.92,  # Space for suptitle
                right=0.97,
                wspace=0.3,
                hspace=0.3
            )

            # Save & close (without tight_layout to preserve our adjustments)
            plt.savefig(os.path.join(self.save_path, file_name))
            plt.savefig(os.path.join(self.save_path, file_name) + ".svg")
            plt.close(fig)

    def clear_metrics(self) -> None:
        """
        Clear the recorded metrics for all agents and all metrics.
        """
        with self._lock:
            self._metrics_history.clear()

    @property
    def metrics_history(self) -> dict:
        """
        Get the entire history of recorded metrics.
        """
        with self._lock:
            return self._metrics_history # {m: {a: dict(es) for a, es in agents.items()}
                    #for m, agents in self._metrics_history.items()}

    def export_metrics(self, file_path: str) -> None:
        """
        Export all recorded metrics to a JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def import_metrics(self, file_path: str) -> None:
        """
        Import metrics from a previously exported JSON file.
        Overwrites any existing metrics.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        self._metrics_history.clear()
        for metric_name, agents in data.items():
            for agent_id, episodes in agents.items():
                for episode_idx_str, values in episodes.items():
                    episode_idx = int(episode_idx_str)
                    self._metrics_history[metric_name][agent_id][episode_idx].extend(values)

    def import_and_plot_focus_metrics(self,
                                      entity: str,
                                      project: str,
                                      run_ids: List[str] = None,
                                      run_filter: dict = None,
                                      plot_save_dir: str = None):
        """
        Import and plot focus metrics from specific WandB runs
        with step-to-episode reconstruction
        """
        focus_metrics = ['entropy', 'value_loss', 'policy_loss', 'return', 'loss']

        # Set save directory
        original_save_path = self.save_path
        if plot_save_dir:
            self.set_save_path(plot_save_dir)

        # Initialize WandB API
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        # Select runs
        selected_runs = []
        for run in runs:
            if run_ids and run.id not in run_ids:
                continue

            if run_filter:
                skip = False
                for key, values in run_filter.items():
                    run_value = getattr(run, key, [])
                    if isinstance(run_value, list):
                        if not any(v in run_value for v in values):
                            skip = True
                            break
                    else:
                        if run_value not in values:
                            skip = True
                            break
                if skip:
                    continue

            selected_runs.append(run)

        print(f"Found {len(selected_runs)} runs matching criteria")

        # Import focus metrics with episode reconstruction
        for run in selected_runs:
            try:
                agent_id = run.config.get('agent')
                data = ast.literal_eval(agent_id)
                agent_id = data.get('agent_type')
                history = run.scan_history()

                # Variables for episode reconstruction
                current_episode = 0
                episode_metrics = {m: [] for m in focus_metrics}
                episode_ended = False

                for row in history:
                    # Detect episode boundaries using return metric
                    if 'return' in row and row['return'] is not None:
                        # Finalize current episode
                        for metric in focus_metrics:
                            if metric == 'return':
                                # Use the actual return value
                                value = row['return']
                            elif episode_metrics[metric]:
                                # Use average for other metrics
                                value = np.mean(episode_metrics[metric])
                            else:
                                continue

                            if metric not in self._metrics_history:
                                self.register_metric(metric)

                            self.record_metric(
                                metric_name=metric,
                                agent_id=agent_id,
                                episode_idx=current_episode,
                                value=value
                            )

                        # Reset for next episode
                        current_episode += 1
                        episode_metrics = {m: [] for m in focus_metrics}
                        episode_ended = True
                    else:
                        episode_ended = False

                    # Collect step-level metrics
                    for metric in focus_metrics:
                        if metric != 'return' and metric in row and isinstance(row[metric], (int, float)):
                            episode_metrics[metric].append(row[metric])

                # Handle last episode if it wasn't finalized
                if not episode_ended and any(episode_metrics.values()):
                    for metric in focus_metrics:
                        if metric == 'return':
                            # No return value for last episode - skip
                            continue
                        elif episode_metrics[metric]:
                            value = np.mean(episode_metrics[metric])
                            if metric not in self._metrics_history:
                                self.register_metric(metric)

                            self.record_metric(
                                metric_name=metric,
                                agent_id=agent_id,
                                episode_idx=current_episode,
                                value=value
                            )

                print(f"Imported {current_episode} episodes from run: {run.name} ({run.id})")

            except Exception as e:
                print(f"Failed to import run {run.id}: {str(e)}")
                traceback.print_exc()

        agent_order = ['GPPO', 'PPO', 'RANDOM']  # Define desired order
        # Generate plots for each focus metric
        for metric in focus_metrics:
            if metric in self._metrics_history:
                print(f"Plotting {metric}...")
                self.plot_metric(
                    metric_name=metric,
                    file_name=metric,
                    x_axis_label="Episodes",
                    y_axis_label=f"{metric} IQM with 95% CI",
                    title=" ",
                    agent_order=agent_order  # Pass the ordering
                )

        # Restore original save path
        if plot_save_dir:
            self.set_save_path(original_save_path)

        # Return final metrics for analysis
        final_results = {}
        for metric in focus_metrics:
            if metric in self._metrics_history:
                final_results[metric] = self.get_final_metrics(metric)

        return final_results

if __name__ == "__main__":
    # Initialize your tracker
    tracker = MetricsTracker()
    results = tracker.import_and_plot_focus_metrics(
        entity="ml_exp",
        project="gppo-drl",
        run_ids=["z2glkm4d", "4pkenkhk", "aw7c443v", "ulyg11gv", "hsszs7co", "qsxyjwqe"],
        plot_save_dir="../results"
    )