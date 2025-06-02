import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os
import pandas as pd
from scipy.stats import t
import logging

class RLAnalyzer:
    """
    Utility for analyzing RL training results and comparing different algorithms.
    """
    def __init__(self):
        """Initialize the RL analyzer."""
        self.results = {}
        self.metrics_of_interest = [
            'avg_reward',
            'served_percentage',
            'avg_epsilon',
            'actor_loss',
            'critic_loss',
            'entropy'
        ]
        self.metric_labels = {
            'avg_reward': 'Average Reward',
            'served_percentage': 'Served Users (%)',
            'avg_epsilon': 'Average Epsilon',
            'actor_loss': 'Actor Loss',
            'critic_loss': 'Critic Loss',
            'entropy': 'Entropy'
        }
    
    def add_result(self, algorithm_name: str, metrics: dict):
        """
        Add a training result to the analyzer.
        
        Args:
            algorithm_name (str): Name of the algorithm.
            metrics (dict): Dictionary of metrics from training.
        """
        self.results[algorithm_name] = metrics
    
    def load_result(self, algorithm_name: str, path: str):
        """
        Load a training result from a file.
        
        Args:
            algorithm_name (str): Name of the algorithm.
            path (str): Path to the metrics file.
        """
        if os.path.exists(path):
            metrics = np.load(path, allow_pickle=True).item()
            self.results[algorithm_name] = metrics
            logging.info(f"Loaded metrics for {algorithm_name}")
        else:
            logging.error(f"Metrics file for {algorithm_name} not found at {path}")
    
    def calculate_convergence_stats(self, window_size=10):
        """
        Calculate convergence statistics for each algorithm.
        
        Args:
            window_size (int): Window size for moving average.
            
        Returns:
            dict: Dictionary of convergence statistics.
        """
        stats = {}
        
        for algo_name, metrics in self.results.items():
            algo_stats = {}
            
            for metric_name in self.metrics_of_interest:
                if metric_name not in metrics or not metrics[metric_name]:
                    continue
                    
                values = np.array(metrics[metric_name])
                
                # Apply moving average
                if len(values) >= window_size:
                    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed = values
                
                # Find point of convergence (when values stabilize)
                if len(smoothed) > 1:
                    # Calculate standard deviation of differences
                    diffs = np.abs(np.diff(smoothed))
                    threshold = np.mean(diffs) * 0.1  # 10% of mean difference
                    
                    # Find first point where differences stay below threshold
                    converged_at = None
                    for i in range(len(diffs)):
                        if np.all(diffs[i:min(i+window_size, len(diffs))] < threshold):
                            converged_at = i
                            break
                    
                    if converged_at is not None:
                        converged_value = smoothed[converged_at]
                    else:
                        converged_at = len(smoothed) - 1
                        converged_value = smoothed[-1]
                else:
                    converged_at = 0
                    converged_value = smoothed[0] if len(smoothed) > 0 else np.nan
                  algo_stats[metric_name] = {
                    'convergence_iteration': converged_at + window_size // 2,  # Adjust for window
                    'convergence_value': converged_value,
                    'final_value': values[-1]
                }
            
            stats[algo_name] = algo_stats
        
        return stats
    
    def plot_training_curves(self, save_path=None, figsize=(15, 10)):
        """
        Plot training curves for all algorithms and metrics.
        
        Args:
            save_path (str): Path to save the figure. If None, figure is displayed.
            figsize (tuple): Figure size.
        """
        num_metrics = len(self.metrics_of_interest)
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
        
        if num_metrics == 1:
            axes = [axes]
        
        # # Generate 100 distinct colors using a continuous colormap
        # colors = plt.cm.get_cmap('hsv', 100)  # or 'nipy_spectral', 'tab20', etc.

        # # Example: get all 100 colors as RGB tuples
        # color_list = [colors(i) for i in range(100)]
        import distinctipy

        # Get 100 perceptually distinct colors as (R, G, B) tuples
        colors = distinctipy.get_colors(100)
        
        for i, metric_name in enumerate(self.metrics_of_interest):
            ax = axes[i]
              for j, (algo_name, metrics) in enumerate(self.results.items()):
                if metric_name in metrics and metrics[metric_name]:
                    values = metrics[metric_name]
                    iterations = np.arange(1, len(values) + 1)
                    
                    ax.plot(iterations, values, label=algo_name, color=colors[j % len(colors)])
            
            ax.set_ylabel(self.metric_labels.get(metric_name, metric_name))
            ax.set_title(f"{self.metric_labels.get(metric_name, metric_name)} vs. Iteration")
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right')
        
        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_convergence_stats(self, save_path=None, figsize=(10, 6)):
        """
        Plot convergence statistics for all algorithms.
        
        Args:
            save_path (str): Path to save the figure. If None, figure is displayed.
            figsize (tuple): Figure size.
        """
        stats = self.calculate_convergence_stats()
        
        # Prepare data for plotting
        algorithms = list(stats.keys())
        metrics = list(self.metrics_of_interest)
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)        # Plot convergence iterations
        convergence_iterations = {}
        for algo in algorithms:
            convergence_iterations[algo] = [stats[algo].get(metric, {}).get('convergence_iteration', np.nan) for metric in metrics]
        
        df_iterations = pd.DataFrame(convergence_iterations, index=[self.metric_labels.get(m, m) for m in metrics])
        
        # Check if there's any data to plot
        if not df_iterations.empty and not df_iterations.isna().all().all():
            df_iterations.plot(kind='bar', ax=axes[0])
            axes[0].set_title('Convergence Iteration by Algorithm and Metric')
            axes[0].set_ylabel('Iteration')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No convergence data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('Convergence Iteration by Algorithm and Metric')
            axes[0].set_ylabel('Iteration')
          # Plot final values
        final_values = {}
        for algo in algorithms:
            final_values[algo] = [stats[algo].get(metric, {}).get('final_value', np.nan) for metric in metrics]
        
        df_values = pd.DataFrame(final_values, index=[self.metric_labels.get(m, m) for m in metrics])
        
        # Check if there's any data to plot
        if not df_values.empty and not df_values.isna().all().all():
            df_values.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Final Values by Algorithm and Metric')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No final value data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('Final Values by Algorithm and Metric')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def compare_algorithms(self, metric_name: str, n_runs=20, confidence=0.95, save_path=None, figsize=(10, 6)):
        """
        Compare multiple runs of different algorithms on a specific metric.
        
        Args:
            metric_name (str): Name of the metric to compare.
            n_runs (int): Number of runs for calculating statistics.
            confidence (float): Confidence level for intervals.
            save_path (str): Path to save the figure. If None, figure is displayed.
            figsize (tuple): Figure size.
        """
        # Check if we have multiple runs data
        algo_metrics = {}
        
        for algo_name, metrics in self.results.items():
            if f"{metric_name}_runs" in metrics:
                run_values = metrics[f"{metric_name}_runs"]
                if len(run_values) >= n_runs:
                    algo_metrics[algo_name] = run_values[:n_runs]
            elif metric_name in metrics:
                # Just use the single run data
                algo_metrics[algo_name] = [metrics[metric_name]]
        
        if not algo_metrics:
            logging.warning(f"No data available for metric '{metric_name}'")
            return
        
        # Calculate statistics
        algo_stats = {}
        for algo_name, run_values in algo_metrics.items():
            # Calculate mean and confidence interval
            mean_values = np.mean(run_values, axis=0)
            std_err = np.std(run_values, axis=0, ddof=1) / np.sqrt(len(run_values))
            t_value = t.ppf((1 + confidence) / 2, len(run_values) - 1)
            margin = t_value * std_err
            
            algo_stats[algo_name] = {
                'mean': mean_values,
                'lower': mean_values - margin,
                'upper': mean_values + margin
            }
        
        # Plot the results
        plt.figure(figsize=figsize)
        
        import distinctipy

        # Get 100 perceptually distinct colors as (R, G, B) tuples
        colors = distinctipy.get_colors(100)
        episodes = None
          for i, (algo_name, stats) in enumerate(algo_stats.items()):
            mean = stats['mean']
            lower = stats['lower']
            upper = stats['upper']
            
            iterations = np.arange(1, len(mean) + 1)
            
            plt.plot(iterations, mean, label=algo_name, color=colors[i % len(colors)])
            plt.fill_between(iterations, lower, upper, alpha=0.2, color=colors[i % len(colors)])
        
        plt.title(f"{self.metric_labels.get(metric_name, metric_name)} Comparison ({n_runs} runs, {confidence*100}% CI)")
        plt.xlabel("Iteration")
        plt.ylabel(self.metric_labels.get(metric_name, metric_name))
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, save_path, window_size=10):
        """
        Generate a comprehensive report of all results.
        
        Args:
            save_path (str): Directory to save the report files.
            window_size (int): Window size for moving average.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Generate training curves
        self.plot_training_curves(save_path=os.path.join(save_path, "training_curves.png"))
        
        # Generate convergence statistics
        self.plot_convergence_stats(save_path=os.path.join(save_path, "convergence_stats.png"))
        
        # Compare algorithms for each metric
        for metric_name in self.metrics_of_interest:
            self.compare_algorithms(metric_name, save_path=os.path.join(save_path, f"{metric_name}_comparison.png"))
        
        # Generate summary table
        stats = self.calculate_convergence_stats(window_size=window_size)
        summary = {algo: {} for algo in stats.keys()}
          for algo_name, algo_stats in stats.items():
            for metric_name, metric_stats in algo_stats.items():
                summary[algo_name][f"{metric_name}_convergence"] = metric_stats['convergence_iteration']
                summary[algo_name][f"{metric_name}_final"] = metric_stats['final_value']
        
        df_summary = pd.DataFrame.from_dict(summary, orient='index')
        df_summary.to_csv(os.path.join(save_path, "summary_stats.csv"))
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>RL Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>RL Training Results Report</h1>
            
            <h2>Summary Statistics</h2>
            {df_summary.to_html()}
            
            <h2>Training Curves</h2>
            <img src="training_curves.png" alt="Training Curves">
            
            <h2>Convergence Statistics</h2>
            <img src="convergence_stats.png" alt="Convergence Statistics">
            
            <h2>Metric Comparisons</h2>
        """
        
        for metric_name in self.metrics_of_interest:
            html_content += f"""
            <h3>{self.metric_labels.get(metric_name, metric_name)}</h3>
            <img src="{metric_name}_comparison.png" alt="{metric_name} Comparison">
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(os.path.join(save_path, "report.html"), 'w') as f:
            f.write(html_content)
        
        logging.info(f"Report generated at {save_path}")
    
    def compare_results(self, result_sets, labels=None, save_path=None):
        """
        Compare multiple RL result sets.
        
        Args:
            result_sets (list): List of result dictionaries from different RL simulations.
            labels (list, optional): Labels for each result set. Defaults to ["Set 1", "Set 2", ...].
            save_path (str, optional): Path to save the comparison plot.
            
        Returns:
            pd.DataFrame: DataFrame with comparison metrics.
        """
        if labels is None:
            labels = [f"Set {i+1}" for i in range(len(result_sets))]
        
        # Create comparison metrics
        metrics = [
            'served_percentage', 
            'avg_epsilon', 
            'avg_utility',
            'execution_time'
        ]
        
        # Create DataFrame for comparison
        comparison_data = {'Metric': metrics}
        
        # Add each result set to the comparison
        for i, results in enumerate(result_sets):
            label = labels[i]
            comparison_data[label] = [results.get(m, np.nan) for m in metrics]
        
        comparison = pd.DataFrame(comparison_data)
        
        # Plot comparison
        if save_path:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
            
            for i, metric in enumerate(metrics):
                ax = axes[i] if len(metrics) > 1 else axes
                
                values = [result_set.get(metric, np.nan) for result_set in result_sets]
                
                ax.bar(labels, values, color=['blue', 'orange', 'green', 'red'][:len(labels)])
                ax.set_title(f'Comparison of {metric}')
                ax.set_ylabel(metric)
                
                # Add values on top of bars
                for j, v in enumerate(values):
                    ax.text(j, v, f'{v:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        return comparison
    
    def analyze_fl_impact(self, metrics_dict, save_path=None):
        """
        Analyze the impact of federated learning on RL performance.
        
        Args:
            metrics_dict (dict): Dictionary mapping FL mode to metrics.
            save_path (str, optional): Path to save the analysis plot.
            
        Returns:
            pd.DataFrame: DataFrame with FL impact analysis.
        """
        # Extract key metrics for each FL mode
        modes = list(metrics_dict.keys())
        metrics = ['served_percentage', 'avg_epsilon', 'avg_reward', 'execution_time']
        
        results = pd.DataFrame(index=modes, columns=metrics)
        
        for mode, data in metrics_dict.items():
            for metric in metrics:
                if metric in data:
                    results.loc[mode, metric] = data[metric]
        
        # Plot FL mode comparison
        if save_path:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
            
            for i, metric in enumerate(metrics):
                ax = axes[i] if len(metrics) > 1 else axes
                
                values = results[metric].values
                
                ax.bar(modes, values)
                ax.set_title(f'Impact of FL mode on {metric}')
                ax.set_ylabel(metric)
                ax.set_xlabel('FL Mode')
                
                # Add values on top of bars
                for j, v in enumerate(values):
                    if not np.isnan(v):
                        ax.text(j, v, f'{v:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        return results
    
    def plot_privacy_vs_qos(self, results_list, labels, save_path=None):
        """
        Plot privacy vs QoS (served percentage) trade-off.
        
        Args:
            results_list (list): List of result dictionaries.
            labels (list): List of labels for each result set.
            save_path (str, optional): Path to save the plot.
        """
        epsilons = [r.get('avg_epsilon', np.nan) for r in results_list]
        served = [r.get('served_percentage', np.nan) for r in results_list]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(epsilons, served, marker='o')
        
        # Label each point
        for i, label in enumerate(labels):            
            plt.annotate(label, (epsilons[i], served[i]), 
                         textcoords='offset points',
                         xytext=(5, 5),
                         ha='center')
        plt.xlabel('Privacy (Avg. Epsilon)')
        plt.ylabel('QoS (Served %)')
        plt.title('Privacy vs QoS Trade-off')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_comparison(self, metrics=None, window_size=10, save_path=None, figsize=(15, 12)):
        """
        Plot comparison of different RL configurations for specified metrics.
        
        Args:
            metrics (list): List of metric names to plot. If None, uses default metrics.
            window_size (int): Window size for moving average smoothing.
            save_path (str, optional): Path to save the plot.
            figsize (tuple): Figure size.
        """
        if not metrics:
            # Default metrics to compare
            metrics = ['avg_reward', 'served_percentage', 'avg_epsilon']
            
        if not self.results:
            logging.warning("No results available for comparison.")
            return
            
        # Plot only the requested metrics
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
        
        # Handle case with only one metric
        if num_metrics == 1:
            axes = [axes]
        
        import distinctipy

        # Get 100 perceptually distinct colors as (R, G, B) tuples
        colors = distinctipy.get_colors(100)
        line_styles = ['-', '--', ':', '-.']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, metric_name in enumerate(metrics):
            ax = axes[i]
              for j, (algo_name, all_metrics) in enumerate(self.results.items()):
                if metric_name in all_metrics and len(all_metrics[metric_name]) > 0:
                    values = all_metrics[metric_name]
                    iterations = np.arange(1, len(values) + 1)
                    
                    # Plot raw values with low alpha
                    ax.plot(iterations, values, alpha=0.3, 
                           color=colors[j % len(colors)])
                    
                    # Plot smoothed values
                    if len(values) > window_size:
                        smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                        ax.plot(iterations, smoothed, 
                               label=algo_name,
                               linewidth=2, 
                               color=colors[j % len(colors)],
                               linestyle=line_styles[j % len(line_styles)],
                               marker=markers[j % len(markers)],
                               markevery=max(1, len(iterations)//10))
                    else:
                        ax.plot(iterations, values, 
                               label=algo_name, 
                               linewidth=2,
                               color=colors[j % len(colors)],
                               linestyle=line_styles[j % len(line_styles)],
                               marker=markers[j % len(markers)],
                               markevery=max(1, len(iterations)//10))
              ax.set_ylabel(self.metric_labels.get(metric_name, metric_name))
            ax.set_title(f"{self.metric_labels.get(metric_name, metric_name)} Comparison")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Comparison plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_learning_curve(self, metrics, window_size=10, save_path=None):
        """
        Plot learning curves from training metrics.
        
        Args:
            metrics (dict): Dictionary of training metrics.
            window_size (int): Window size for moving average smoothing.
            save_path (str, optional): Path to save the plot.
        """
        if not metrics:
            logging.warning("No metrics provided for learning curve plot.")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot reward
        rewards = metrics.get('rewards', [])
        if rewards:
            axes[0].plot(rewards, alpha=0.3, color='blue')
            # Smoothed curve            if len(rewards) > window_size:
                smooth_rewards = pd.Series(rewards).rolling(window=window_size).mean()
                axes[0].plot(smooth_rewards, linewidth=2, color='blue')
            axes[0].set_title('Training Reward')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True)
        
        # Plot served percentage
        served = metrics.get('served_percentages', [])
        if served:
            axes[1].plot(served, alpha=0.3, color='green')
            if len(served) > window_size:
                smooth_served = pd.Series(served).rolling(window=window_size).mean()
                axes[1].plot(smooth_served, linewidth=2, color='green')
            axes[1].set_title('Served Users Percentage')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Served %')
            axes[1].grid(True)
        
        # Plot epsilon
        epsilons = metrics.get('epsilons', [])
        if epsilons:
            axes[2].plot(epsilons, alpha=0.3, color='red')
            if len(epsilons) > window_size:
                smooth_eps = pd.Series(epsilons).rolling(window=window_size).mean()
                axes[2].plot(smooth_eps, linewidth=2, color='red')
            axes[2].set_title('Average Epsilon (Privacy)')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Epsilon')
            axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
