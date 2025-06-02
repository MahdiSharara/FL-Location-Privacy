import logging
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from utilities.common_setup import (
    setup_privacy_maximization,
    setup_privacy_satisfaction,
    setup_privacy_cancellation,
    load_system_configuration
)
from utilities.results_manager import ResultsManager
from data_structures import User, Node, Link
from generate_data import generate_users, generate_nodes, generate_links
from utilities.generate_channel_gains import generate_channel_gains
from rl.trainer_improved import InfiniteHorizonTrainer
from rl.analyzer import RLAnalyzer
from data_structures import obj

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_scenario(users, nodes, links, num_RBs, num_MCS, channel_gains, noise_power, 
                      nrTBSMatrix, gamma_th, config, scenario, save_dir):
    """
    Run a single RL scenario and save the results.
    
    Args:
        users: Dictionary of User objects
        nodes: Dictionary of Node objects
        links: Dictionary of Link objects
        num_RBs: Number of resource blocks
        num_MCS: Number of MCS schemes
        channel_gains: Channel gain matrix
        noise_power: Noise power
        nrTBSMatrix: TBS matrix
        gamma_th: SINR thresholds
        config: Configuration dictionary
        scenario: Scenario dictionary
        save_dir: Directory to save results
        
    Returns:
        Training metrics dictionary
    """
    # Prepare users for the scenario
    scenario_users = {uid: user.copy() if hasattr(user, 'copy') else user for uid, user in users.items()}
    scenario["setup"](scenario_users)    # Create the trainer
    from rl.resource_allocator import ResourceAllocator
    
    resource_allocator = ResourceAllocator(
        scenario_users, nodes, links, num_RBs, num_MCS, 
        channel_gains, nrTBSMatrix, noise_power, gamma_th, config
    )
    
    trainer = InfiniteHorizonTrainer(
        config=config,
        users=scenario_users, 
        nodes=nodes, 
        links=links, 
        resource_allocator=resource_allocator,
        device='cpu'
    )    # Train the agents
    logging.info(f"Starting training for scenario: {scenario['name']}")
    
    if scenario.get("use_infinite_horizon", False) and scenario.get("max_iterations") is not None:
        # Use infinite horizon training - continuous, on-policy, single-step updates
        logging.info(f"Using infinite horizon training with {scenario['max_iterations']} iterations")
        training_metrics = trainer.train_infinite_horizon(
            total_iterations=scenario["max_iterations"],
            update_frequency=scenario.get("update_frequency", 32),
            save_frequency=scenario.get("save_frequency", 1000),
            eval_frequency=scenario.get("eval_frequency", 500),
            save_path=save_dir
        )
    else:
        # Use infinite horizon training as default (removing legacy episodic mode)
        logging.info(f"Using default infinite horizon training with 10000 iterations")
        training_metrics = trainer.train_infinite_horizon(
            total_iterations=10000,
            update_frequency=32,
            save_frequency=1000,
            eval_frequency=500,
            save_path=save_dir
        )    
    # Evaluate the agents
    eval_metrics = trainer.evaluate_performance(num_eval_episodes=10)
    logging.info(f"Evaluation results: {eval_metrics}")
    
    # Save the trained models
    scenario_dir = os.path.join(save_dir, scenario['name'].replace(' ', '_'))
    trainer.save_models(scenario_dir)
    
    return training_metrics

def main():
    # Fix random seed for reproducibility if needed
    np.random.seed(42)
    
    # Load configuration using consolidated utility
    config, nrTBSMatrix, _ = load_system_configuration()
    f = config.get("frequency", 2)  # Default frequency in GHz
    num_RBs = config["n_RBs"]
    num_MCS = config["n_MCS"]
    gamma_th = config["gamma_th"]
      # Define RL scenarios - Updated for infinite horizon training
    rl_scenarios = [
        {
            "name": "RL-Local-Both-FL",
            "setup": setup_privacy_maximization,
            "fl_mode": "both",  # Both actor and critic use FL
            "max_iterations": 1000,  # Replaced training_episodes * steps_per_episode
            "use_infinite_horizon": True,  # Enable infinite horizon training
            "fl_rounds": 5,
            "cancel_privacy": False,
        },
        {
            "name": "RL-Local-Critic-FL",
            "setup": setup_privacy_maximization,
            "fl_mode": "critic",  # Only critic uses FL
            "max_iterations": 1000,  # Replaced training_episodes * steps_per_episode
            "use_infinite_horizon": True,  # Enable infinite horizon training
            "fl_rounds": 5,
            "cancel_privacy": False,
        },
        {
            "name": "RL-Local-No-FL",
            "setup": setup_privacy_maximization,
            "fl_mode": "none",  # No FL at all
            "max_iterations": 1000,  # Replaced training_episodes * steps_per_episode
            "use_infinite_horizon": True,  # Enable infinite horizon training
            "fl_rounds": 5,
            "cancel_privacy": False,
        },
        {
            "name": "RL-Privacy-Cancellation",
            "setup": setup_privacy_cancellation,
            "fl_mode": "both",  # Both actor and critic use FL
            "max_iterations": 1000,  # Replaced training_episodes * steps_per_episode
            "use_infinite_horizon": True,  # Enable infinite horizon training
            "fl_rounds": 5,
            "cancel_privacy": True,
        },
    ]
    
    # Set up the environment
    num_users = 10
    num_nodes = 15
    num_links = 0  # Will be auto-generated as needed
    
    # Generate the network
    users = generate_users(num_users)
    nodes = generate_nodes(num_nodes)
    links = generate_links(num_links, nodes)
    
    # Calculate channel gains
    d_real = obj.calculate_distance_between_groups(users, nodes, use_fake_distance=False)
    channel_gains, noise_power = generate_channel_gains(d_real, f, num_RBs)
      # Create save directory for results using results manager
    results_manager = ResultsManager()
    save_dir = results_manager.create_experiment_directory("rl_scenarios")
    
    # Create analyzer for results
    analyzer = RLAnalyzer()
    
    # Run each scenario
    for scenario in rl_scenarios:
        start_time = time.time()
        
        training_metrics = run_single_scenario(
            users, nodes, links, num_RBs, num_MCS, 
            channel_gains, noise_power, nrTBSMatrix, gamma_th, 
            config, scenario, save_dir
        )
        
        # Add results to analyzer
        analyzer.add_result(scenario["name"], training_metrics)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Scenario {scenario['name']} completed in {elapsed_time:.2f} seconds")
    
    # Generate analysis report
    report_dir = os.path.join(save_dir, "analysis")
    os.makedirs(report_dir, exist_ok=True)
    analyzer.generate_report(report_dir)
    
    # Show some plots directly
    analyzer.plot_training_curves()
    analyzer.plot_convergence_stats()
    plt.show()

if __name__ == "__main__":
    main()
