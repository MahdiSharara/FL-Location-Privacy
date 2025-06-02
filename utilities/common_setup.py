"""
Common setup functions and configuration loading.
Consolidates frequently used setup patterns across the codebase.
"""

import logging
import os
from typing import Dict, Optional, Tuple
from data_structures import User, Node, Link
from generate_data import generate_users, generate_nodes, generate_links
from utilities.load_json import load_config
from utilities.nrTBS import load_nrTBSMatrix
from utilities.generate_channel_gains import generate_channel_gains
from utilities.update_user_epsilon import update_user_epsilon
from data_structures import obj

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_privacy_maximization(users: Dict[int, User]) -> Dict[int, User]:
    """Set up users for privacy maximization (default behavior)"""
    return users


def setup_privacy_satisfaction(users: Dict[int, User]) -> Dict[int, User]:
    """Set up users to use their maximum epsilon value"""
    for user in users.values():
        user.set_epsilon_generate_fake_location(user.max_epsilon)
    return users


def setup_privacy_cancellation(users: Dict[int, User]) -> Dict[int, User]:
    """Set up users to share their real location (no privacy)"""
    update_user_epsilon(users, None, cancel_privacy=True)
    return users


def load_system_configuration(config_path: Optional[str] = None) -> Tuple[dict, dict, dict]:
    """
    Load system configuration and TBS matrix.
    
    Args:
        config_path: Path to config file. If None, uses default 'config.json'
        
    Returns:
        Tuple of (config, nrTBSMatrix, nrTBS_dict)
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    
    config = load_config(config_path)
    nrTBSMatrix, nrTBS_dict = load_nrTBSMatrix(config['nrTBSMatrix_file'], config['nrTBSMatrix_variable'])
    
    return config, nrTBSMatrix, nrTBS_dict


def create_simulation_environment(num_users: int, num_nodes: int, num_links: int, 
                                  config: dict, frequency: float = 2.4) -> Tuple[Dict, Dict, Dict, any, float]:
    """
    Create a complete simulation environment with users, nodes, links, and channel gains.
    
    Args:
        num_users: Number of users to generate
        num_nodes: Number of nodes to generate  
        num_links: Number of links to generate
        config: Configuration dictionary
        frequency: Operating frequency in GHz
        
    Returns:
        Tuple of (users, nodes, links, channel_gains, noise_power)
    """
    # Generate network topology
    users = generate_users(num_users)
    nodes = generate_nodes(num_nodes)
    links = generate_links(num_links, nodes)
    
    # Calculate channel gains
    d_real = obj.calculate_distance_between_groups(users, nodes, use_fake_distance=False)
    channel_gains, noise_power = generate_channel_gains(
        d_real=d_real,
        f=frequency,
        n_RBs=config['n_RBs'],
        noise_figure_db=config['noise_figure'],
        bandwidth=config['B_RB_Hz']
    )
    
    logger.info(f"Created simulation environment: {num_users} users, {num_nodes} nodes, {num_links} links")
    
    return users, nodes, links, channel_gains, noise_power


def get_common_rl_parameters(config: dict) -> dict:
    """
    Extract common RL parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of common RL parameters
    """
    return {
        'num_RBs': config['n_RBs'],
        'num_MCS': config['n_MCS'],
        'gamma_th': config['gamma_th'],
        'noise_figure': config['noise_figure'],
        'bandwidth': config['B_RB_Hz']
    }


def setup_rl_trainer_environment(users: Dict[int, User], nodes: Dict[int, Node], 
                                 links: Dict[int, Link], config: dict, 
                                 channel_gains: any, noise_power: float, 
                                 nrTBS_dict: dict, gamma_th: any, device: str = 'cpu'):
    """
    Set up the complete RL training environment.
    
    Args:
        users: Dictionary of User objects
        nodes: Dictionary of Node objects
        links: Dictionary of Link objects
        config: Configuration dictionary
        channel_gains: Channel gain matrix
        noise_power: Noise power
        nrTBS_dict: TBS dictionary
        gamma_th: SINR thresholds
        device: Training device ('cpu' or 'cuda')
        
    Returns:
        Tuple of (resource_allocator, trainer)
    """
    try:
        from rl.resource_allocator import ResourceAllocator
        from rl.trainer_improved import InfiniteHorizonTrainer
        
        # Create resource allocator
        resource_allocator = ResourceAllocator(
            users, nodes, links, config['n_RBs'], config['n_MCS'],
            channel_gains, nrTBS_dict, noise_power, gamma_th, config
        )
        
        # Create trainer
        trainer = InfiniteHorizonTrainer(
            config=config,
            users=users,
            nodes=nodes,
            links=links,
            resource_allocator=resource_allocator,
            device=device
        )
        
        return resource_allocator, trainer
        
    except ImportError as e:
        logger.error(f"RL components not available: {e}")
        return None, None


# Common scenario definitions
PRIVACY_SCENARIOS = {
    "privacy_maximization": {
        "setup": setup_privacy_maximization,
        "description": "Users maximize their privacy (default behavior)"
    },
    "privacy_satisfaction": {
        "setup": setup_privacy_satisfaction,
        "description": "Users use their maximum epsilon value"
    },
    "privacy_cancellation": {
        "setup": setup_privacy_cancellation,
        "description": "Users share real location (no privacy)"
    }
}


def get_privacy_setup_function(scenario_name: str):
    """Get the privacy setup function for a given scenario name."""
    scenario = PRIVACY_SCENARIOS.get(scenario_name)
    if scenario is None:
        logger.warning(f"Unknown privacy scenario: {scenario_name}. Using privacy_maximization.")
        return setup_privacy_maximization
    return scenario["setup"]
