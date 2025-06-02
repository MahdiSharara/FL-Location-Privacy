import logging
import copy
import numpy as np
import time
import os
import multiprocessing
from multiprocessing import Pool
from data_structures import obj
from generate_data import generate_users, generate_nodes, generate_links
from plotting import plot_user_and_node_positions
from utilities.load_json import load_config
from utilities.generate_channel_gains import generate_channel_gains
from utilities.update_user_parameters import update_user_parameters
from utilities.nrTBS import load_nrTBSMatrix
from utilities.analyze_results_tools import analyze_results, analyze_results_multiple
from scipy.stats import sem, t

# Try to import RL components, but don't fail if they're not available
try:
    import torch
    from rl.trainer_improved import InfiniteHorizonTrainer
    from rl.resource_allocator import ResourceAllocator
    RL_AVAILABLE = True
except ImportError:
    logging.warning("RL components not available. RL scenarios will be skipped.")
    RL_AVAILABLE = False


def simulate_single_run(args, scenarios, num_RBs, num_MCS, nrTBSMatrix, f):
    """
    Simulates a single run of the privacy algorithm using RL.

    Args:
        args (tuple): A tuple containing the number of users, run index, number of nodes, links, RBs, and MCS.
        scenarios (list): List of scenarios to simulate.
        num_RBs (int): Number of resource blocks.
        num_MCS (int): Number of modulation and coding schemes.
        nrTBSMatrix (np.ndarray): NR TBS Matrix for throughput calculations.
        f (float): Frequency in GHz.

    Returns:
        dict: Results of the simulation for different scenarios.
    """
    num_users, run, num_nodes, num_links = args

    # Generate users, nodes, and links
    users = generate_users(num_users)
    nodes = generate_nodes(num_nodes)
    links = generate_links(num_links, nodes)

    # Calculate channel gains
    d_real = obj.calculate_distance_between_groups(users, nodes, use_fake_distance=False)
    channel_gains, noise_power = generate_channel_gains(d_real, f, num_RBs)

    # Load configuration
    config = load_config('config.json')
    gamma_th = config['gamma_th']    # Run each scenario
    results = {}
    for scenario in scenarios:
        logging.info(f"nUsers: {num_users}, run: {run+1}, phase: {scenario['name']}")

        # Prepare users for the scenario
        scenario_users = copy.deepcopy(users)
        
        # Apply specific setup if needed
        if "setup" in scenario and scenario["setup"] is not None:
            scenario["setup"](scenario_users)        # Handle RL and Non-RL scenarios
        if not RL_AVAILABLE:
            logging.warning(f"Skipping scenario {scenario['name']} as RL components are not available")
            results[scenario["name"]] = {}
            continue
              # For scenarios that use RL
        if scenario.get("use_rl", True):
            # Execute RL-based approach
            results[scenario["name"]] = run_rl_solution(
                scenario_users, nodes, links, num_RBs, num_MCS, channel_gains, noise_power,
                nrTBSMatrix, gamma_th, config, 
                scenario.get("fl_mode", "both"), 
                scenario.get("training_episodes", 50),
                scenario.get("steps_per_episode", 10),
                scenario.get("fl_rounds", 5),
                scenario.get("cancel_privacy", False),
                num_users, run,
                scenario.get("use_infinite_horizon", False),
                scenario.get("max_iterations", None)
            )
        else:
            # For non-RL scenarios (like Privacy-Cancellation)
            # These use a direct heuristic approach without RL training
            if scenario.get("cancel_privacy", False):
                # Set all users to have infinite epsilon (no privacy)
                for user_id, user in scenario_users.items():
                    user.set_epsilon_generate_fake_location(float('inf'))
                
            # Create a resource allocator without RL
            resource_allocator = ResourceAllocator(
                scenario_users, nodes, links, num_RBs, num_MCS, 
                channel_gains, nrTBSMatrix, noise_power, gamma_th, config
            )
            
            # Evaluate using the direct heuristic
            start_time = time.time()
            
            # Run the allocator directly
            allocator_result = resource_allocator.allocate_resources_heuristic()
            
            # Calculate metrics
            execution_time = time.time() - start_time
            avg_epsilon = sum(user.assigned_epsilon for user in scenario_users.values()) / len(scenario_users)
            served_percentage = sum(1 for user in scenario_users.values() if user.is_served) / len(scenario_users) * 100
            
            # Analyze the results and calculate metrics
            metrics = analyze_results(scenario_users, nodes, links)
            metrics.update(allocator_result)
            metrics.update({
                "execution_time": execution_time,
                "avg_epsilon": avg_epsilon,
                "served_percentage": served_percentage,
            })
            
            results[scenario["name"]] = metrics

    return {
        "num_users": num_users,
        "run": run + 1,
        **results,
    }

def is_valid_result(result):
    """
    Checks if a result is valid by ensuring all numerical values are finite.

    Args:
        result (dict): The result dictionary to validate.

    Returns:
        bool: True if the result is valid, False otherwise.
    """
    return all(
        isinstance(value, (int, float, np.ndarray)) and np.isfinite(value).all()
        for key, value in result.items() if isinstance(value, (int, float, np.ndarray))
    )

def run_simulation(num_runs, list_num_users, num_nodes, num_links, scenarios, num_RBs, num_MCS, nrTBSMatrix, f, parallel=True):
    """
    Runs the simulation for multiple runs and user configurations.

    Args:
        num_runs (int): Number of runs for each configuration.
        list_num_users (list): List of user counts to simulate.
        num_nodes (int): Number of nodes.
        num_links (int): Number of links.
        scenarios (list): List of scenarios to simulate.
        num_RBs (int): Number of resource blocks.
        num_MCS (int): Number of modulation and coding schemes.
        nrTBSMatrix (np.ndarray): NR TBS Matrix for throughput calculations.
        f (float): Frequency in GHz.
        parallel (bool): Whether to run simulations in parallel.

    Returns:
        dict: Aggregated results for all configurations.
    """
    args = [(num_users, run, num_nodes, num_links) for num_users in list_num_users for run in range(num_runs)]
    results = []

    if parallel:
        with Pool() as pool:
            try:
                results = pool.starmap(
                    simulate_single_run,
                    [(arg, scenarios, num_RBs, num_MCS, nrTBSMatrix, f) for arg in args]
                )
            except Exception as e:
                logging.error(f"Error during parallel execution: {e}")
    else:
        for arg in args:
            result = simulate_single_run(arg, scenarios, num_RBs, num_MCS, nrTBSMatrix, f)
            if result is not None:
                results.append(result)

    # Aggregate results
    aggregated_results = {}
    for num_users in list_num_users:
        scenario_results = {scenario["name"]: [] for scenario in scenarios}

        # Collect results for each scenario
        for res in results:
            if res["num_users"] == num_users:
                for scenario in scenarios:
                    scenario_name = scenario["name"]
                    scenario_results[scenario_name].append(res.get(scenario_name))

        # Validate and aggregate results for each scenario
        aggregated_results[num_users] = {}
        for scenario_name, scenario_res in scenario_results.items():
            valid_results = []
            invalid_results = []

            # Separate valid and invalid results in one pass
            for res in scenario_res:
                if is_valid_result(res):
                    valid_results.append(res)
                else:
                    invalid_results.append(res)

            # Log invalid results
            if invalid_results:
                logging.warning(f"Invalid results for '{scenario_name}' with {num_users} users: {invalid_results}")

            # Handle valid results
            if not valid_results:
                logging.warning(f"No valid results for '{scenario_name}' with {num_users} users.")
                aggregated_results[num_users][scenario_name] = None
            else:
                aggregated_results[num_users][scenario_name] = analyze_results_multiple(valid_results)

    return aggregated_results

def run_rl_solution(
    users, nodes, links, num_RBs, num_MCS, channel_gains, noise_power,
    nrTBSMatrix, gamma_th, config, fl_mode, training_episodes,
    steps_per_episode, fl_rounds, cancel_privacy, num_users, run,
    use_infinite_horizon=False, max_iterations=None
):
    """
    Run the RL-based solution for location privacy.

    Args:
        users (dict): Dictionary of users.
        nodes (dict): Dictionary of nodes.
        links (dict): Dictionary of links.
        num_RBs (int): Number of resource blocks.
        num_MCS (int): Number of modulation and coding schemes.
        channel_gains (np.ndarray): Channel gains matrix.
        noise_power (float): Noise power.
        nrTBSMatrix (np.ndarray): NR TBS Matrix for throughput calculations.
        gamma_th (list): SINR thresholds for each MCS level.
        config (dict): Configuration dictionary.
        fl_mode (str): Federated learning mode ('both', 'actor', 'critic', 'none').
        training_episodes (int): Number of training episodes (for episodic training).
        steps_per_episode (int): Number of steps per episode (for episodic training).
        fl_rounds (int): Number of federated learning rounds.
        cancel_privacy (bool): Flag to indicate if privacy cancellation is applied.
        num_users (int): Number of users in the simulation.
        run (int): Current simulation run index.
        use_infinite_horizon (bool): Whether to use infinite horizon training.
        max_iterations (int): Maximum iterations for infinite horizon training.

    Returns:
        dict: Results of the simulation.
    """# Start measuring execution time
    start_time = time.time()
      # If privacy is cancelled, set all users to have infinite epsilon (no privacy)
    if cancel_privacy:
        for user_id, user in users.items():
            user.set_epsilon_generate_fake_location(float('inf'))
    
    # Train RL agents
    resource_allocator = ResourceAllocator(
        users, nodes, links, num_RBs, num_MCS, 
        channel_gains, nrTBSMatrix, noise_power, gamma_th, config
    )
    
    trainer = InfiniteHorizonTrainer(
        config=config,
        users=users, 
        nodes=nodes, 
        links=links, 
        resource_allocator=resource_allocator,
        device='cpu'    )
      
    # Training phase
    train_start_time = time.time()
    
    if use_infinite_horizon and max_iterations is not None:
        # Use infinite horizon training - continuous, on-policy, single-step updates
        logging.info(f"nUsers: {num_users}, run: {run+1}, Starting infinite horizon training with {max_iterations} iterations")
        training_metrics = trainer.train_infinite_horizon(
            total_iterations=max_iterations,
            update_frequency=32,
            save_frequency=1000,
            eval_frequency=500
        )
    else:
        # Use infinite horizon training as default (removing legacy episodic mode)
        logging.info(f"nUsers: {num_users}, run: {run+1}, Using default infinite horizon training with 10000 iterations")
        training_metrics = trainer.train_infinite_horizon(
            total_iterations=10000,
            update_frequency=32,
            save_frequency=1000,
            eval_frequency=500
        )
    
    train_time = time.time() - train_start_time
    training_mode = "Infinite Horizon" if use_infinite_horizon else "Episodic"
    logging.info(f"nUsers: {num_users}, run: {run+1}, RL {training_mode} Training time: {train_time:.2f}s")
      # Evaluation phase
    eval_start_time = time.time()
    eval_metrics = trainer.evaluate_performance(num_eval_episodes=10)
    eval_time = time.time() - eval_start_time
    logging.info(f"nUsers: {num_users}, run: {run+1}, RL Evaluation time: {eval_time:.2f}s")
    
    # Get the allocation results based on the trained policy
    allocator = trainer.resource_allocator
    solution = allocator.allocate()
    
    # Update user parameters based on the allocation
    update_user_parameters(users, solution, nrTBSMatrix)
      # Calculate total execution time
    total_execution_time = time.time() - start_time
    # Get results without needing to pass x and z
    results = analyze_results(users, nodes, links)
    
    # Add RL-specific metrics
    results.update({
        "execution_time_total": total_execution_time,
        "execution_time_training": train_time,
        "execution_time_evaluation": eval_time,
        "avg_reward": eval_metrics["avg_reward"],
        "served_percentage": eval_metrics["served_percentage"],
        "avg_epsilon": eval_metrics["avg_epsilon"],
        "training_metrics": training_metrics,
    })
    
    return results

# MILP solution removed - Only RL solution is available now

