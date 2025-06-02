#!/usr/bin/env python
# test_rl_integration.py - Integration testing for RL-based location privacy solution

import logging
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
import copy
from utilities.common_setup import (
    load_system_configuration,
    setup_rl_trainer_environment,
    setup_privacy_maximization,
    setup_privacy_satisfaction,
    setup_privacy_cancellation
)
from utilities.results_manager import ResultsManager
from data_structures import User, Node, Link, obj
from generate_data import generate_users, generate_nodes, generate_links
from utilities.generate_channel_gains import generate_channel_gains
from rl.trainer_improved import InfiniteHorizonTrainer
from rl.analyzer import RLAnalyzer
from rl.resource_allocator import ResourceAllocator
from simulation import run_rl_solution, RL_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rl_single_scenario(federate_mode='both', differential_privacy=None, use_infinite_horizon=True):
    """Test the RL pipeline with a single scenario setup"""
    training_mode = "infinite horizon" if use_infinite_horizon else "episodic"
    logger.info(f"Testing RL integration with federate_mode={federate_mode}, differential_privacy={differential_privacy}, training_mode={training_mode}")
    
    # Load configuration using consolidated utility
    config, nrTBSMatrix, nrTBS_dict = load_system_configuration()
    
    # Test parameters
    num_users = 5
    num_nodes = 3
    num_links = 3
    num_RBs = config['n_RBs']
    num_MCS = config['n_MCS']
    gamma_th = config['gamma_th']
    
    # Create test data
    users = generate_users(num_users)
    nodes = generate_nodes(num_nodes)
    links = generate_links(num_links, nodes)
      # Calculate channel gains
    f = 2.4  # GHz
    # Calculate the distance matrix first
    d_real = obj.calculate_distance_between_groups(users, nodes, use_fake_distance=False)
    # Call generate_channel_gains with proper arguments
    channelGains, noise_power = generate_channel_gains(
        d_real=d_real, 
        f=f, 
        n_RBs=num_RBs,
        noise_figure_db=config['noise_figure'],
        bandwidth=config['B_RB_Hz']
    )
    
    # Run RL solution
    scenario = {
        "name": f"Test-RL-{federate_mode}-FL",
        "use_rl": True,
        "federate_mode": federate_mode,
        "differential_privacy": differential_privacy,
        "episodes": 5,  # Small number for testing
        "batch_size": 4,
        "max_noise_radius": 1000  # meters
    }    # Run the test
    start_time = time.time()
    if use_infinite_horizon:
        # Test infinite horizon training with fewer iterations for quick testing
        result = run_rl_solution(
            users, nodes, links, num_RBs, num_MCS, 
            channelGains, noise_power, nrTBS_dict, gamma_th, config,
            scenario.get("federate_mode", "both"),
            50,  # training_episodes (not used in infinite horizon)
            1,   # steps_per_episode (not used in infinite horizon)
            3,   # fl_rounds
            False,  # cancel_privacy
            len(users),  # num_users
            0,   # run index
            True,  # use_infinite_horizon
            100   # max_iterations (smaller for testing)
        )
    else:
        # Test episodic training
        result = run_rl_solution(
            users, nodes, links, num_RBs, num_MCS, 
            channelGains, noise_power, nrTBS_dict, gamma_th, config,
            scenario.get("federate_mode", "both"),
            scenario.get("episodes", 5),  # training_episodes
            4,  # steps_per_episode
            3,  # fl_rounds
            False,  # cancel_privacy
            len(users),  # num_users
            0,  # run index
            False,  # use_infinite_horizon
            None   # max_iterations
        )
    elapsed = time.time() - start_time
      # Validate results
    assert result is not None, "RL solution returned None result"
    assert "percentage_served_users" in result, "Missing percentage_served_users in results" 
    assert "avg_epsilon" in result, "Missing avg_epsilon in results"
    assert "training_metrics" in result, "Missing training metrics in results"
    
    logger.info(f"Test completed in {elapsed:.2f}s")
    logger.info(f"Results: served={result['percentage_served_users']:.2f}%, " + 
               f"avg_epsilon={result['avg_epsilon']:.4f}")
    
    return result

def test_compare_fl_modes():
    """Compare different federated learning modes"""
    logger.info("Testing comparison of FL modes")
    
    # Run different configurations using infinite horizon training
    results = {}
    
    # Test with different FL modes using infinite horizon
    results["both_infinite"] = test_rl_single_scenario(federate_mode='both', use_infinite_horizon=True)
    results["critic_infinite"] = test_rl_single_scenario(federate_mode='critic', use_infinite_horizon=True)
    results["none_infinite"] = test_rl_single_scenario(federate_mode='none', use_infinite_horizon=True)
    
    # Test with differential privacy using infinite horizon
    results["both_dp_infinite"] = test_rl_single_scenario(federate_mode='both', differential_privacy=0.5, use_infinite_horizon=True)
    
    # Optional: Also test episodic mode for comparison
    results["both_episodic"] = test_rl_single_scenario(federate_mode='both', use_infinite_horizon=False)
      # Create analyzer and add results
    analyzer = RLAnalyzer()
    for mode, result in results.items():
        analyzer.add_result(mode, result["training_metrics"])
    
    # Plot comparison
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    analyzer.plot_comparison(metrics=["avg_reward", "served_percentage", "avg_epsilon"], 
                            save_path=os.path.join(output_dir, "fl_comparison.png"))
    
    logger.info(f"Comparison plots saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Run tests - focusing on infinite horizon training
    logger.info("Testing infinite horizon training...")
    test_rl_single_scenario(use_infinite_horizon=True)
    
    logger.info("Testing episodic training for comparison...")
    test_rl_single_scenario(use_infinite_horizon=False)
    
    logger.info("Comparing different FL modes...")
    test_compare_fl_modes()
    
    logger.info("All tests completed successfully")
