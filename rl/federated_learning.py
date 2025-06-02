import numpy as np
import torch
import copy
from typing import Dict, List, Tuple
import logging
import torch.nn.functional as F

class FederatedLearner:
    """
    Implements Federated Learning for the RL agents.
    """
    def __init__(self, num_users: int, federate_actor: bool = True, federate_critic: bool = True, 
                 dp_eps: float = None, dp_clip: float = 1.0):
        """
        Initialize the Federated Learning system.
        
        Args:
            num_users (int): Number of users/clients in the system.
            federate_actor (bool): Whether to federate the actor model.
            federate_critic (bool): Whether to federate the critic model.
            dp_eps (float): Differential privacy epsilon. If None, no DP is applied.
            dp_clip (float): Gradient clipping value for DP.
        """
        self.num_users = num_users
        self.federate_actor = federate_actor
        self.federate_critic = federate_critic
        self.dp_eps = dp_eps
        self.dp_clip = dp_clip
        self.metrics = {'actor': [], 'critic': []}
    
    def aggregate(self, agents: Dict[int, object]):
        """
        Perform federated aggregation of agent models.
        
        Args:
            agents (Dict[int, object]): Dictionary of agent objects.
            
        Returns:
            dict: Aggregated model state dictionaries.
        """
        # Get model state dictionaries
        state_dicts = {user_id: agent.get_state_dict() for user_id, agent in agents.items()}
        
        # Initialize aggregated state dict with the structure of the first agent
        if not state_dicts:
            return None
        
        first_id = list(state_dicts.keys())[0]
        aggregated_state = copy.deepcopy(state_dicts[first_id])
        
        # Aggregate actor if requested
        if self.federate_actor:
            actor_params = self._aggregate_component('actor', state_dicts)
            aggregated_state['actor'] = actor_params
            aggregated_state['actor_optimizer'] = state_dicts[first_id]['actor_optimizer']  # Keep optimizer state
        
        # Aggregate critic if requested
        if self.federate_critic:
            critic_params = self._aggregate_component('critic', state_dicts)
            aggregated_state['critic'] = critic_params
            aggregated_state['critic_optimizer'] = state_dicts[first_id]['critic_optimizer']  # Keep optimizer state
        
        return aggregated_state
    
    def _aggregate_component(self, component_name: str, state_dicts: Dict[int, dict]):
        """
        Aggregate parameters for a specific component (actor or critic).
        
        Args:
            component_name (str): Component name ('actor' or 'critic').
            state_dicts (Dict[int, dict]): Dictionary of agent state dictionaries.
            
        Returns:
            dict: Aggregated parameters for the component.
        """
        # Initialize with the structure of the first agent's component
        first_id = list(state_dicts.keys())[0]
        component_dict = copy.deepcopy(state_dicts[first_id][component_name])
        
        # Reset all parameters to zero for averaging
        for param_name, param in component_dict.items():
            component_dict[param_name] = torch.zeros_like(param)
        
        # Collect gradients and apply differential privacy if configured
        for user_id, state_dict in state_dicts.items():
            user_params = state_dict[component_name]
            
            # Apply DP if configured
            if self.dp_eps is not None:
                user_params = self._apply_differential_privacy(user_params, self.dp_eps, self.dp_clip)
            
            # Add to the aggregate
            for param_name, param in user_params.items():
                component_dict[param_name] += param
        
        # Average the parameters
        for param_name in component_dict:
            component_dict[param_name] /= len(state_dicts)
        
        return component_dict
    
    def _apply_differential_privacy(self, params: dict, epsilon: float, clip_value: float):
        """
        Apply differential privacy to parameters using the Gaussian mechanism.
        
        Args:
            params (dict): Parameters dictionary.
            epsilon (float): Privacy parameter epsilon.
            clip_value (float): Clipping value for gradients.
            
        Returns:
            dict: Parameters with DP noise applied.
        """
        # Create a copy to avoid modifying the original
        dp_params = copy.deepcopy(params)
        
        # Calculate sensitivity (after clipping)
        sensitivity = clip_value * np.sqrt(2 * np.log(1.25 / 0.05)) / epsilon
        
        # Apply clipping and add noise to each parameter
        for param_name, param in dp_params.items():
            # Clip the parameter values
            with torch.no_grad():
                param_norm = torch.norm(param)
                scale = min(1.0, clip_value / (param_norm + 1e-8))
                clipped_param = param * scale
                
                # Add calibrated gaussian noise
                noise = torch.randn_like(clipped_param) * sensitivity
                dp_params[param_name] = clipped_param + noise
        
        return dp_params
    
    def apply_updates(self, agents: Dict[int, object], aggregated_state: dict):
        """
        Apply the aggregated model updates to all agents.
        
        Args:
            agents (Dict[int, object]): Dictionary of agent objects.
            aggregated_state (dict): Aggregated model state dictionary.
        """
        for agent in agents.values():
            agent.set_state_dict(aggregated_state)
    
    def log_metrics(self, round_num: int, aggregated_state: dict):
        """
        Log metrics about the federated learning process.
        
        Args:
            round_num (int): Current round number.
            aggregated_state (dict): Aggregated model state dictionary.
        """        # Calculate some metrics about the aggregated model (e.g., parameter norms)
        actor_norm = sum(torch.norm(param).item() for param in aggregated_state['actor'].values())
        critic_norm = sum(torch.norm(param).item() for param in aggregated_state['critic'].values())
        self.metrics['actor'].append(actor_norm)
        self.metrics['critic'].append(critic_norm)
        
        logging.info(f"FL Round {round_num}: Actor norm = {actor_norm:.4f}, Critic norm = {critic_norm:.4f}")
    
    def get_metrics(self):
        """
        Get the recorded metrics.
        
        Returns:
            dict: Dictionary of metric histories.
        """
        return self.metrics
        
    def get_state(self):
        """
        Get the current state of the federated learner.
        
        Returns:
            dict: State dictionary.
        """
        return {
            'metrics': self.metrics,
            'num_users': self.num_users,
            'federate_actor': self.federate_actor,
            'federate_critic': self.federate_critic,
            'dp_eps': self.dp_eps,
            'dp_clip': self.dp_clip
        }
    
    def set_state(self, state):
        """
        Set the state of the federated learner.
        
        Args:
            state (dict): State dictionary.
        """
        if state is None:
            return
            
        if 'metrics' in state:
            self.metrics = state['metrics']
        if 'num_users' in state:
            self.num_users = state['num_users']
        if 'federate_actor' in state:
            self.federate_actor = state['federate_actor']
        if 'federate_critic' in state:
            self.federate_critic = state['federate_critic']
        if 'dp_eps' in state:
            self.dp_eps = state['dp_eps']
        if 'dp_clip' in state:
            self.dp_clip = state['dp_clip']
