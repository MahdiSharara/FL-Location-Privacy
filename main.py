from simulation import run_simulation
from plotting import plot_results, plot_plotly, plot_dash, metrics_to_plot
from utilities.common_setup import (
    setup_privacy_maximization, 
    setup_privacy_satisfaction, 
    setup_privacy_cancellation,
    load_system_configuration
)
from utilities.results_manager import save_multiple_run_results, ResultsManager
from utilities.load_json import load_config
from utilities.nrTBS import load_nrTBSMatrix
import threading
import time

fixRandom = False
if fixRandom:
    # Fix the random seed for reproducibility
    import random
    import numpy as np
    seed = 42  # Choose any number you want to use as your fixed seed
    random.seed(seed)
    np.random.seed(seed)

# Only RL scenarios are implemented

# Define RL scenarios - Updated for infinite horizon training
rl_scenarios = [
    {
        "name": "RL-Local-Both-FL",
        "use_rl": True,
        "fl_mode": "both",  # Both actor and critic use FL
        "max_iterations": 320,  # Increased to get 10 data points (320/32 = 10)
        "use_infinite_horizon": True,  # Enable infinite horizon training
        "fl_rounds": 1,
        "cancel_privacy": False,
    },
    {
        "name": "RL-Local-Critic-FL",
        "use_rl": True,
        "fl_mode": "critic",  # Only critic uses FL, actor is local
        "max_iterations": 320,  # Increased to get 10 data points (320/32 = 10)
        "use_infinite_horizon": True,  # Enable infinite horizon training
        "fl_rounds": 1,
        "cancel_privacy": False,
    },
    {
        "name": "RL-Local-No-FL",
        "use_rl": True,
        "fl_mode": "none",  # No FL, both actor and critic are local
        "max_iterations": 320,  # Increased to get 10 data points (320/32 = 10)
        "use_infinite_horizon": True,  # Enable infinite horizon training
        "fl_rounds": 1,
        "cancel_privacy": False,
    },
]

# Define no-privacy scenario (uses direct heuristic approach with real locations)
no_privacy_scenario = {
    "name": "Privacy-Cancellation",
    "use_rl": False,  # Don't use RL
    "fl_mode": "none", # No FL needed
    "cancel_privacy": True,  # Use real locations
}

# Use RL scenarios and the no-privacy scenario
scenarios = rl_scenarios.copy()
scenarios.append(no_privacy_scenario)
if __name__ == "__main__":
    # Launch a timer to measure the execution time of the script
    start_time = time.time()

    # Load configuration using consolidated utility
    config, nrTBSMatrix, _ = load_system_configuration()
    f = config.get("frequency", 2)  # Default frequency in GHz if not specified in config
    num_RBs = config["n_RBs"]
    num_MCS = config["n_MCS"]

    # Simulation parameters
    num_runs = 2  # Number of simulation runs for each user count
    list_num_users = [5, 10, 15, 20, 25]  # Number of users to test
    num_nodes = 15  # Number of nodes
    num_links = 0  # Max Number of links: only add if necessary link < max links
    parallel = False  # Set to True for parallel execution, False for sequential

    # Run the simulation
    aggregated_results = run_simulation(
        num_runs, list_num_users, num_nodes, num_links, scenarios, num_RBs, num_MCS, nrTBSMatrix, f, parallel=parallel
    )
    time_taken = time.time() - start_time
    print(f"Time taken for simulation: {time_taken:.2f} seconds, excluding plotting")

    # Check if we want to analyze RL metrics specifically
    analyze_rl_results = True
    if analyze_rl_results:
        try:
            from rl.analyzer import RLAnalyzer
            
            # Create an RL analyzer
            analyzer = RLAnalyzer()
            
            # Add RL scenario results
            for scenario in rl_scenarios:
                # Extract training metrics for each user count
                for num_users in list_num_users:
                    if num_users in aggregated_results:
                        scenario_name = scenario["name"]
                        if scenario_name in aggregated_results[num_users]:
                            result_data = aggregated_results[num_users][scenario_name]
                            if "training_metrics" in result_data:
                                analyzer.add_result(f"{scenario_name}-{num_users}users", 
                                                   result_data["training_metrics"])
              # Generate reports using results manager
            from datetime import datetime
            import os

            # Create results manager and setup experiment directory
            results_manager = ResultsManager()
            save_dir = results_manager.create_experiment_directory("main_analysis")

            # Full path for the report file
            report_path = os.path.join(save_dir, "_rl_analysis_results")
            os.makedirs(report_path, exist_ok=True)
              # Generate and save the report using analyzer
            analyzer.generate_report(report_path)
            print(f"RL analysis report generated at: {report_path}")
            
            # Save aggregated results using results manager
            results_manager.save_results(aggregated_results, save_dir, "aggregated_results.json")
            
            # Plot specific comparisons
            analyzer.plot_training_curves()
            analyzer.plot_convergence_stats()
            
            # Compare algorithms on specific metrics
            for metric_name in ["avg_reward", "served_percentage", "avg_epsilon"]:
                analyzer.compare_algorithms(metric_name,n_runs=num_runs)
                
        except ImportError:
            print("RL analyzer not available. Skipping RL-specific analysis.")
    
    # Plot the results using Matplotlib
    plot_results(aggregated_results, scenarios, metrics_to_plot)

    # Plot the results using Plotly for interactive visualization
    try:
        plot_plotly(aggregated_results, scenarios, metrics_to_plot)
        
        # Launch Dash for browser-based interactive visualization in a separate thread
        dash_thread = threading.Thread(target=plot_dash, args=(aggregated_results, scenarios, metrics_to_plot))
        dash_thread.daemon = True  # Ensure the thread exits when the main program exits
        dash_thread.start()
    except Exception as e:
        print(f"Error launching interactive plots: {e}")

    # Keep Matplotlib plots open after the program terminates
    import matplotlib.pyplot as plt
    plt.show(block=True)
    # Wait for the Dash thread to finish (it won't, but this keeps the main thread alive)
    #dash_thread.join()