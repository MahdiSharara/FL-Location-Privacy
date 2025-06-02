import numpy as np
from typing import Dict, List, Tuple
from data_structures import User, Node, Link, obj
from scipy.stats import t
from utilities.load_json import load_config
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = load_config('config.json')
metric_bounds = {
    # ...existing metrics...
    "average_delay_per_user": (0, None),  # No upper bound
    "average_delay_violation": (0, None),  # No upper bound
    "average_throughput_per_user": (0, None),  # No upper bound
    "Total_throughput": (0, None),  # No upper bound
    "percentage_served_users": (0, 100),
    "percentage_average_RBs_load": (0, 100),
    "percentage_average_processing_utilization": (0, 100),
    "average_epsilon_variation": (0, None),  # No bounds
    "percentage_average_link_utilization": (0, 100),
    "average_link_utilization": (0, 1),  # No upper bound
    "percentage_link_usage": (0, 100),
    "served_users": (0, None),
    # New execution time metrics
    "execution_time_total": (0, None),  # No upper bound
    "execution_time_step0": (0, None),  # No upper bound 
    "execution_time_step1": (0, None),  # No upper bound
}


def calculate_confidence_interval(values: List[float], bounds: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate the mean and confidence interval for a list of values, respecting bounds.

    Args:
        values (List[float]): List of values.
        bounds (Tuple[float, float]): Lower and upper bounds for the metric.

    Returns:
        Tuple[float, Tuple[float, float]]: Mean and confidence interval (lower, upper).
    """
    # Convert values to ensure they are numeric
    numeric_values = []
    for val in values:
        # Handle numpy arrays by taking their mean
        if isinstance(val, np.ndarray):
            if val.size > 0:
                numeric_values.append(np.mean(val))
        elif isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
            numeric_values.append(val)
    
    # If no valid values, return default values
    if not numeric_values:
        return 0.0, (0.0, 0.0)
    
    mean = np.mean(numeric_values)
    
    # If only one value, return the mean with zero margin
    if len(numeric_values) == 1:
        return mean, (mean, mean)
    
    std_err = np.std(numeric_values, ddof=1) / np.sqrt(len(numeric_values))
    margin = t.ppf(0.975, len(numeric_values) - 1) * std_err  # 95% confidence interval

    lower_bound, upper_bound = bounds
    ci_lower = max(lower_bound, mean - margin) if lower_bound is not None else mean - margin
    ci_upper = min(upper_bound, mean + margin) if upper_bound is not None else mean + margin

    return mean, (ci_lower, ci_upper)

def analyze_results(users: Dict[int, User], nodes: Dict[int, Node], links: Dict[int, Link]):
    """
    Analyze the results based on the user/node/link properties for RL solutions only.
    Handles routing decisions from real BS through fake BS to server.

    Args:
        users (Dict[int, User]): Dictionary of User objects.
        nodes (Dict[int, Node]): Dictionary of Node objects.
        links (Dict[int, Link]): Dictionary of Link objects.

    Returns:
        Dict[str, float]: A dictionary containing the computed metrics.
    """
    total_users = len(users)
    total_links = len(links)
    # Get data attribute from the users
    data = np.array([user.rate_requirement for user in users.values()])
    
    # User metrics - based on user properties set by the RL solution
    served_users = [user for user in users.values() if user.is_served and user.associated_server is not None]
    total_served_users = len(served_users)
    total_delay = sum(user.achieved_delay or 0 for user in served_users)
    
    # Calculate the violated delay among the users who were served but their delay requirement was violated
    total_delay_violation = sum(user.achieved_delay - user.delay_requirement for user in served_users if user.achieved_delay > user.delay_requirement)
    # Calculate the number of users whose delay was violated
    total_users_with_violated_delay = len([user for user in served_users if user.achieved_delay > user.delay_requirement])
    # Find the average among the users who were served but their delay requirement was violated
    average_delay_violation = total_delay_violation / total_users_with_violated_delay if total_users_with_violated_delay > 0 else 0
    
    total_throughput = sum(user.achieved_throughput or 0 for user in served_users)
    total_assigned_RBs = sum(user.assigned_RBs or 0 for user in served_users)
    
    # Calculate epsilon variation (privacy metric)
    epsilon_variation_sum = sum(
        abs(user.assigned_epsilon - user.max_epsilon) / total_served_users
        for user in users.values() if user.assigned_epsilon is not None and user.assigned_epsilon != float('inf') 
    ) if total_served_users > 0 else 0

    # Node metrics
    # Calculate the total utilized processing resources from all users
    total_used_processing_resources = sum(
        user.rate_requirement for user in served_users if user.associated_server is not None
    )
    
    # total_processing_utilization = sum(
    #     node.processing_capacity * sum(
    #         user.rate_requirement for user in served_users if user.associated_server == node.id
    #     ) * (node.processing_capacity > 0) / np.where(node.processing_capacity == 0, 1, node.processing_capacity) 
    #     for node in nodes.values() if any(user.associated_server == node.id for user in served_users)
    # )
    aggregated_links_capacity = 0
    aggregated_links_usage = 0
    # Link metrics - Calculate utilization directly from user properties
    link_utilizations = []
    
    # Map to track which links are used by which users
    link_usage = {link_id: set() for link_id in links.keys()}
    
    # In RL solution, user data flows through:
    # Real BS -> Fake BS -> Server
    # But we only have access to the real BS and server in user properties
    # We need to infer the fake BS location from the user's fake_location
    
    # Identify which users are using which links based on their BS and server assignments
    for user in served_users:
        if user.associated_BS is not None and user.associated_server is not None:
            # If using a basic model where we assume direct links:
            if user.associated_BS != user.associated_server:
                # First identify the fake BS node (closest node to user's fake location)
                fake_bs_id = None
                min_distance = float('inf')
                
                # Find the node (fake BS) closest to the user's fake location
                if user.fake_location is not None:
                    for node_id, node in nodes.items():
                        if node.node_type == 'G':  # Ground stations are BSs
                            distance = obj.calculate_distance(user, node, use_fake_distance=True)
                            if distance < min_distance:
                                min_distance = distance
                                fake_bs_id = node_id
                
                # If we couldn't determine the fake BS, fall back to direct link
                if fake_bs_id is None or fake_bs_id == user.associated_BS or fake_bs_id == user.associated_server:
                    # Direct link between real BS and server
                    for link_id, link in links.items():
                        if ((link.node_1.id == user.associated_BS and link.node_2.id == user.associated_server) or 
                            (link.node_1.id == user.associated_server and link.node_2.id == user.associated_BS)):
                            link_usage[link_id].add(user.id)
                else:
                    # Two-hop routing: real BS -> fake BS -> server
                    # First hop: real BS -> fake BS
                    for link_id, link in links.items():
                        if ((link.node_1.id == user.associated_BS and link.node_2.id == fake_bs_id) or 
                            (link.node_1.id == fake_bs_id and link.node_2.id == user.associated_BS)):
                            link_usage[link_id].add(user.id)
                    
                    # Second hop: fake BS -> server
                    for link_id, link in links.items():
                        if ((link.node_1.id == fake_bs_id and link.node_2.id == user.associated_server) or 
                            (link.node_1.id == user.associated_server and link.node_2.id == fake_bs_id)):
                            link_usage[link_id].add(user.id)
    
    # Calculate link utilization based on the identified usage
    for link_id, link in links.items():
        using_users = link_usage[link_id]
        # Sum the data requirements of all users using this link
        data_flow = sum(users[user_id].rate_requirement for user_id in using_users)
        aggregated_links_usage += data_flow
        aggregated_links_capacity += link.link_capacity
        
        # Calculate link utilization (data flow / capacity)
        if link.link_capacity > 0:
            link_utilizations.append(data_flow / link.link_capacity)
        else:
            link_utilizations.append(0)

    average_link_utilization = np.mean(link_utilizations) if link_utilizations else 0
    total_number_of_Nodes = len(nodes)
    total_number_of_RBs = total_number_of_Nodes * config['n_RBs']
    #total Processing resources of all nodes together
    total_processing_resources = sum(node.processing_capacity for node in nodes.values())
    # Compute averages and percentages
    average_delay_per_user = total_delay / total_served_users if total_served_users > 0 else 0  # in seconds
    average_throughput_per_user = total_throughput / total_served_users if total_served_users > 0 else 0  # in Mbps
    percentage_served_users = (total_served_users / total_users) * 100 if total_users > 0 else 0
    served_users = total_served_users
    percentage_average_RBs_load = total_assigned_RBs / total_number_of_RBs * 100
    percentage_average_processing_utilization = total_used_processing_resources * 100 / total_processing_resources if total_processing_resources > 0 else 0
    average_epsilon_variation_per_user = epsilon_variation_sum / total_users if total_users > 0 else 0
    percentage_average_link_utilization = average_link_utilization*100  if total_links > 0 else 0  # Re-enable this line
    percentage_link_usage = aggregated_links_usage*100 / aggregated_links_capacity if aggregated_links_capacity > 0 else 0
    # Return results as a dictionary
    return {
        "average_delay_per_user": average_delay_per_user,  # in seconds
        "average_throughput_per_user": average_throughput_per_user,  # in Mbps
        "Total_throughput": total_throughput,  # in Mbps
        "percentage_served_users": percentage_served_users,
        "percentage_average_RBs_load": percentage_average_RBs_load,
        "percentage_average_processing_utilization": percentage_average_processing_utilization,
        "average_epsilon_variation": average_epsilon_variation_per_user,
        "percentage_average_link_utilization": percentage_average_link_utilization,
        "average_link_utilization": average_link_utilization,
        "percentage_served_users": percentage_served_users,
        "percentage_link_usage": percentage_link_usage,
        "served_users": served_users,
        "average_delay_violation": average_delay_violation,
    }

def analyze_results_multiple(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Analyze multiple sets of results and calculate mean and confidence intervals.

    Args:
        runs (List[Dict[str, float]]): List of result dictionaries from multiple runs.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the mean and confidence interval for each metric.
    """
    # Collect all metrics
    metrics = runs[0].keys()
    aggregated_results = {metric: [] for metric in metrics}

    # Aggregate results from all runs
    for run in runs:
        for metric, value in run.items():
            # Skip complex metrics like training_metrics which are dictionaries
            if not isinstance(value, (int, float, np.ndarray)) or (isinstance(value, dict) and metric == "training_metrics"):
                continue
            aggregated_results[metric].append(value)

    final_results = {}
    # Apply bounds to confidence intervals during calculation
    for metric, values in aggregated_results.items():
        # Skip empty metrics or metrics with non-numeric values
        if not values or not all(isinstance(v, (int, float, np.ndarray)) for v in values):
            continue
            
        bounds = metric_bounds.get(metric, (None, None))
        mean, confidence_interval = calculate_confidence_interval(values, bounds)

        # Log the calculated confidence intervals
        logging.info(f"Metric: {metric}, Mean: {mean:.2f}, CI: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")

        # Convert numpy types to Python native types for JSON serialization
        if isinstance(mean, np.number):
            mean = mean.item()  # Convert numpy scalar to native Python type
        
        ci_lower, ci_upper = confidence_interval
        if isinstance(ci_lower, np.number):
            ci_lower = ci_lower.item()
        if isinstance(ci_upper, np.number):
            ci_upper = ci_upper.item()
        
        final_results[metric] = {
            "mean": mean,
            "confidence_interval": (ci_lower, ci_upper)
        }
        
    # Handle special metrics like training_metrics separately
    if "training_metrics" in metrics and any("training_metrics" in run for run in runs):
        # Convert any numpy types in training_metrics to Python native types
        training_metrics = runs[0].get("training_metrics", {})
        final_results["training_metrics"] = convert_numpy_to_python_types(training_metrics)
        
    return final_results

def convert_numpy_to_python_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: The object to convert.
        
    Returns:
        An object with all numpy types converted to Python native types.
    """
    if isinstance(obj, np.number):
        return obj.item()  # Convert numpy scalar to native Python type
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python_types(item) for item in obj]
    else:
        return obj

    return final_results

# Example usage
if __name__ == "__main__":
    # Example: Load users, nodes, links (replace with actual data)
    users = {}  # Replace with generated users
    nodes = {}  # Replace with generated nodes
    links = {}  # Replace with generated links

    results = analyze_results(users, nodes, links)
    for metric, value in results.items():
        print(f"{metric}: {value}")

    # Example: Simulated results from multiple runs
    runs = [
        {
            "average_delay_per_user": 10.5,  # Updated to match returned metrics
            "average_throughput_per_user": 50.2,
            "Total_throughput": 1000,  # Updated to match returned metrics
            "percentage_served_users": 95.0,
            "percentage_average_RBs_load": 20.1,
            "percentage_average_processing_utilization": 75.3,
            "average_epsilon_variation": 0.05,
            "percentage_average_link_utilization": 80.0,
            "average_link_utilization": 0.8,
            "percentage_link_usage": 85.0,  # Added to match returned metrics
            "served_users": 95,            
            "average_delay_violation": 0.1,
        },
        {
            "average_delay_per_user": 11.0,
            "average_throughput_per_user": 48.7,
            "Total_throughput": 980,
            "percentage_served_users": 94.5,
            "percentage_average_RBs_load": 19.8,
            "percentage_average_processing_utilization": 76.1,
            "average_epsilon_variation": 0.06,
            "percentage_average_link_utilization": 82.0,
            "average_link_utilization": 0.82,
            "percentage_link_usage": 87.0,
            "served_users": 94,
            "average_delay_violation": 0.2,
        },
    ]

    # Analyze results from multiple runs
    results = analyze_results_multiple(runs)
    for metric, stats in results.items():
        print(f"{metric}: mean = {stats['mean']}, confidence interval = ±{stats['confidence_interval']}")