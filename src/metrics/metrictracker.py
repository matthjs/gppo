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
                 n_bootstrap: int = 1000,
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
                    smoothing_window: int = 1,
                    ribbon: bool = True,
                    err_bars: bool = False) -> None:
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


if __name__ == "__main__":
    # Quick testing maybe move this to test file.
    tracker = MetricsTracker(n_bootstrap=500)
    # Simulate logging rewards for 3 agents over 100 episodes with 5 runs each
    for agent in ['agentA', 'agentB', 'agentC']:
        for run in range(5):  # simulate 5 independent runs
            for episode in range(100):
                reward = np.random.normal(loc=episode * 0.1 + run, scale=1.0)  # example reward
                tracker.record_metric('return', agent_id=agent, episode_idx=episode, value=reward)

    # Compute and print IQM and 95% CI at episode 50 for agentA
    iqm, low, high = tracker.get_iqm_ci('return', 'agentA', 50)
    print(f"Episode 50 IQM for agentA: {iqm:.3f} (95% CI: [{low:.3f}, {high:.3f}])")

    # Save a plot of the return metric
    tracker.plot_metric('return', file_name='return_plot.png', y_axis_label="return")
