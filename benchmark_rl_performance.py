#!/usr/bin/env python
# benchmark_rl_performance.py - Evaluate RL performance across scenarios

import logging
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from utilities.common_setup import load_system_configuration, setup_rl_trainer_environment
from utilities.results_manager import ResultsManager
from data_structures import User, Node, Link
from generate_data import generate_users, generate_nodes, generate_links
from utilities.generate_channel_gains import generate_channel_gains
from rl.trainer_improved import InfiniteHorizonTrainer
from rl.analyzer import RLAnalyzer
from rl.resource_allocator import ResourceAllocator
from simulation import run_rl_solution
import multiprocessing
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark(user_counts=[10, 20, 30], 
                 node_counts=[3, 5, 10], 
                 federate_modes=['both', 'critic', 'none'],
                 dp_options=[None, 0.5, 1.0],
                 runs_per_config=3,
                 episodes=200,
                 batch_size=64,
                 parallel=True):
    """
    Run comprehensive benchmarks on RL performance.
    
    Args:
        user_counts: List of user counts to test
        node_counts: List of node counts to test
        federate_modes: List of federated learning modes
        dp_options: List of differential privacy epsilon values (None for no DP)
        runs_per_config: Number of runs per configuration
        episodes: Number of episodes per run
        batch_size: Batch size for training
        parallel: Whether to run in parallel
        
    Returns:
        pd.DataFrame: Benchmark results
    """    # Load configuration using consolidated utility
    config, nrTBSMatrix, _ = load_system_configuration()
    
    # Prepare benchmark configurations
    benchmarks = []
    for user_count in user_counts:
        for node_count in node_counts:
            for federate_mode in federate_modes:
                for dp_epsilon in dp_options:
                    # Skip unreasonable combinations
                    if user_count < 10 and federate_mode != 'none':
                        continue  # Federated learning doesn't make sense with too few users
                        
                    for run in range(runs_per_config):
                        benchmarks.append({
                            'user_count': user_count,
                            'node_count': node_count,
                            'link_count': node_count * 2,  # Assuming ~2 links per node
                            'federate_mode': federate_mode,
                            'dp_epsilon': dp_epsilon,
                            'run': run
                        })
    
    logger.info(f"Prepared {len(benchmarks)} benchmark configurations")
    
    # Run benchmarks
    results = []
    
    if parallel and multiprocessing.cpu_count() > 1:
        # Parallel execution
        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4))
        results = pool.map(run_single_benchmark, [(b, config, nrTBSMatrix, episodes, batch_size) for b in benchmarks])
        pool.close()
        pool.join()
    else:
        # Sequential execution
        for benchmark in benchmarks:
            result = run_single_benchmark((benchmark, config, nrTBSMatrix, episodes, batch_size))
            results.append(result)
    
    # Combine results into DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(output_dir, f"rl_benchmark_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    
    logger.info(f"Benchmark results saved to {csv_path}")
    
    # Generate summary visualizations
    generate_benchmark_visualizations(df_results, output_dir, timestamp)
    
    return df_results

def run_single_benchmark(args):
    """
    Run a single benchmark configuration.
    
    Args:
        args: Tuple of (benchmark_config, global_config, nrTBSMatrix, episodes, batch_size)
        
    Returns:
        dict: Benchmark results
    """
    benchmark, config, nrTBSMatrix, episodes, batch_size = args
    
    logger.info(f"Running benchmark: {benchmark}")
    
    # Generate test data
    users = generate_users(benchmark['user_count'])
    nodes = generate_nodes(benchmark['node_count'])
    links = generate_links(benchmark['link_count'], nodes)
    
    # Fixed parameters
    num_RBs = config['n_RBs']
    num_MCS = config['n_MCS'] 
    gamma_th = config['gamma_th']
    
    # Calculate channel gains and noise power
    f = 2.4  # GHz
    channel_gains = generate_channel_gains(users, nodes, f)
    noise_power = 10 ** ((config['noise_spectral_density'] + 
                        config['noise_figure'] + 
                        10 * np.log10(config['B_RB_Hz'])) / 10)
    
    # Create scenario config
    scenario = {
        "name": f"Benchmark-{benchmark['user_count']}u-{benchmark['node_count']}n-{benchmark['federate_mode']}-dp{benchmark['dp_epsilon']}",
        "use_rl": True,
        "federate_mode": benchmark['federate_mode'],
        "differential_privacy": benchmark['dp_epsilon'],
        "episodes": episodes,
        "batch_size": batch_size,
        "max_noise_radius": 1000  # meters
    }
    
    # Run RL solution
    start_time = time.time()
    try:
        result = run_rl_solution(
            users, nodes, links, num_RBs, num_MCS, 
            channel_gains, noise_power, nrTBSMatrix, gamma_th, config, scenario
        )
        
        # Extract key metrics
        metrics = {
            'user_count': benchmark['user_count'],
            'node_count': benchmark['node_count'],
            'federate_mode': benchmark['federate_mode'],
            'dp_epsilon': benchmark['dp_epsilon'],
            'run': benchmark['run'],
            'served_percentage': result['served_percentage'],
            'avg_epsilon': result['avg_epsilon'],
            'execution_time': result['execution_time'],
            'success': True
        }
        
        # Add convergence metrics
        training_metrics = result.get('metrics', {})
        if training_metrics:
            # Use last 10% of episodes for stability metrics
            n = len(training_metrics.get('avg_reward', []))
            if n > 0:
                last_idx = max(1, int(n * 0.1))
                metrics['final_reward'] = np.mean(training_metrics['avg_reward'][-last_idx:])
                metrics['reward_std'] = np.std(training_metrics['avg_reward'][-last_idx:])
            
            # Calculate convergence episode (when performance stabilizes)
            # Defined as when the std of reward over a window is less than 10% of mean
            if 'avg_reward' in training_metrics and len(training_metrics['avg_reward']) > 10:
                window_size = 10
                rewards = np.array(training_metrics['avg_reward'])
                for i in range(len(rewards) - window_size):
                    window = rewards[i:i+window_size]
                    if np.std(window) < 0.1 * np.abs(np.mean(window)):
                        metrics['convergence_episode'] = i
                        break
    except Exception as e:
        logger.error(f"Error in benchmark {benchmark}: {e}")
        metrics = {
            'user_count': benchmark['user_count'],
            'node_count': benchmark['node_count'],
            'federate_mode': benchmark['federate_mode'],
            'dp_epsilon': benchmark['dp_epsilon'],
            'run': benchmark['run'],
            'success': False,
            'error': str(e)
        }
    
    elapsed = time.time() - start_time
    logger.info(f"Completed benchmark in {elapsed:.2f}s: {benchmark}")
    
    return metrics

def generate_benchmark_visualizations(df, output_dir, timestamp):
    """
    Generate visualizations from benchmark results.
    
    Args:
        df (pd.DataFrame): Benchmark results
        output_dir (str): Output directory
        timestamp (str): Timestamp for filenames
    """
    # Filter only successful benchmarks
    df_success = df[df['success'] == True]
    
    if df_success.empty:
        logger.warning("No successful benchmarks to visualize")
        return
    
    # 1. Impact of user count on performance by federate_mode
    plt.figure(figsize=(12, 8))
    for federate_mode in df_success['federate_mode'].unique():
        df_mode = df_success[df_success['federate_mode'] == federate_mode]
        if not df_mode.empty:
            means = df_mode.groupby('user_count')['served_percentage'].mean()
            stds = df_mode.groupby('user_count')['served_percentage'].std()
            plt.errorbar(means.index, means, yerr=stds, label=f'FL Mode: {federate_mode}', 
                        marker='o', capsize=5)
    
    plt.title('Impact of User Count on Served Percentage by FL Mode')
    plt.xlabel('Number of Users')
    plt.ylabel('Served Users (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"user_count_impact_{timestamp}.png"))
    plt.close()
    
    # 2. Privacy (epsilon) vs QoS (served percentage) trade-off
    plt.figure(figsize=(10, 8))
    
    # Group by federate_mode
    for federate_mode in df_success['federate_mode'].unique():
        df_mode = df_success[df_success['federate_mode'] == federate_mode]
        if not df_mode.empty:
            plt.scatter(df_mode['avg_epsilon'], df_mode['served_percentage'], 
                       alpha=0.7, label=f'FL Mode: {federate_mode}')
    
    plt.title('Privacy vs QoS Trade-off')
    plt.xlabel('Privacy (Avg. Epsilon)')
    plt.ylabel('QoS (Served %)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"privacy_qos_tradeoff_{timestamp}.png"))
    plt.close()
    
    # 3. Execution time comparison
    plt.figure(figsize=(12, 8))
    
    # Create user_count + federate_mode combined factor for grouping
    df_success['config'] = df_success['user_count'].astype(str) + '_' + df_success['federate_mode']
    
    # Sort by mean execution time
    configs = df_success.groupby('config')['execution_time'].mean().sort_values().index
    
    means = []
    stds = []
    for config in configs:
        means.append(df_success[df_success['config'] == config]['execution_time'].mean())
        stds.append(df_success[df_success['config'] == config]['execution_time'].std())
    
    plt.bar(range(len(configs)), means, yerr=stds, capsize=5)
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    plt.title('Execution Time Comparison')
    plt.xlabel('Configuration (Users_FLMode)')
    plt.ylabel('Execution Time (s)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"execution_time_{timestamp}.png"))
    plt.close()
    
    # 4. Heatmap of served percentage by user_count and node_count
    pivot = df_success.pivot_table(
        values='served_percentage', 
        index='user_count', 
        columns='node_count', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, cmap='viridis')
    plt.colorbar(label='Served Percentage (%)')
    plt.title('Impact of Network Size on Served Percentage')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Users')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.iloc[i, j]):
                plt.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha='center', va='center', color='white')
    
    plt.savefig(os.path.join(output_dir, f"size_impact_heatmap_{timestamp}.png"))
    plt.close()
    
    # Create summary report
    summary = {
        'timestamp': timestamp,
        'total_benchmarks': len(df),
        'successful_benchmarks': len(df_success),
        'failed_benchmarks': len(df) - len(df_success),
        'avg_served_percentage': df_success['served_percentage'].mean(),
        'avg_epsilon': df_success['avg_epsilon'].mean(),
        'avg_execution_time': df_success['execution_time'].mean()
    }
    
    # Add FL mode comparison
    for mode in df_success['federate_mode'].unique():
        df_mode = df_success[df_success['federate_mode'] == mode]
        summary[f'{mode}_avg_served'] = df_mode['served_percentage'].mean()
        summary[f'{mode}_avg_epsilon'] = df_mode['avg_epsilon'].mean()
    
    # Save summary
    with open(os.path.join(output_dir, f"summary_{timestamp}.txt"), 'w') as f:
        f.write("Benchmark Summary\n")
        f.write("=================\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run RL performance benchmarks')
    parser.add_argument('--users', type=int, nargs='+', default=[10, 20, 30],
                        help='User counts to test')
    parser.add_argument('--nodes', type=int, nargs='+', default=[3, 5, 10],
                        help='Node counts to test')
    parser.add_argument('--modes', type=str, nargs='+', default=['both', 'critic', 'none'],
                        help='FL modes to test')
    parser.add_argument('--dp', type=float, nargs='+', default=[None, 0.5],
                        help='DP epsilon values (use -1 for None)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Runs per configuration')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Episodes per run')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel execution')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Process DP options (replace -1 with None)
    dp_options = [None if dp == -1 else dp for dp in args.dp]
    
    run_benchmark(
        user_counts=args.users,
        node_counts=args.nodes,
        federate_modes=args.modes,
        dp_options=dp_options,
        runs_per_config=args.runs,
        episodes=args.episodes,
        batch_size=args.batch_size,
        parallel=not args.no_parallel
    )
    
    logger.info("Benchmark completed")
