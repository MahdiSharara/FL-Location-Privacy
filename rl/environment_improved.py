import torch
import numpy as np
from data_structures import User, Node, Link, obj
from typing import Dict, List, Tuple

class UserEnvImproved:
    """
    Improved environment for a single user's reinforcement learning agent.
    This environment simulates the decision process for location privacy with
    enhanced state representation including information about other users.
    """
    def __init__(self, user: User, nodes: Dict[int, Node], links: Dict[int, Link], 
                 all_users: Dict[int, User], config: dict):
        """
        Initialize the user environment.
        
        Args:
            user (User): The user object.
            nodes (Dict[int, Node]): Dictionary of node objects.
            links (Dict[int, Link]): Dictionary of link objects.
            all_users (Dict[int, User]): Dictionary of all users for collective information.
            config (dict): Configuration dictionary.
        """
        self.user = user
        self.nodes = nodes
        self.links = links
        self.all_users = all_users
        self.config = config
        self.deltaF = config['deltaF']
        self.x_y_location_range = config['x_y_location_range']
        
        # Keep track of the original real location
        self.original_location = user.real_location.copy()
        
        # Enhanced state space dimension with collective user information:
        # [real_location(2D), rate_req, delay_req, distances_to_nodes, 
        #  throughput, delay, is_served, collective_info(8D)]
        # collective_info: [min_rate, max_rate, mean_rate, std_rate, 
        #                   min_delay, max_delay, mean_delay, std_delay]
        self.collective_info_dim = 8
        self.state_dim = 2 + 2 + len(nodes) + 2 + 1 + self.collective_info_dim
        
        # To store history for evaluation
        self.history = []
        self.reset()
        
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            np.ndarray: Initial state observation.
        """
        # Reset the user to its original state
        self.user.real_location = self.original_location.copy()
        self.user.fake_location = None
        self.user.is_served = False
        self.user.associated_BS = None
        self.user.associated_server = None
        self.user.achieved_throughput = None
        self.user.achieved_delay = None
        
        # Clear history
        self.history = []
        
        return self._get_state()
    
    def _get_collective_user_info(self):
        """
        Get normalized collective information about all other users.
        
        Returns:
            np.ndarray: Normalized collective user statistics.
        """
        # Get other users (excluding current user)
        other_users = [u for u_id, u in self.all_users.items() if u_id != self.user.id]
        
        if not other_users:
            # If no other users, return zeros
            return np.zeros(self.collective_info_dim)
        
        # Extract rate and delay requirements from other users
        rate_reqs = [u.rate_requirement for u in other_users]
        delay_reqs = [u.delay_requirement for u in other_users]
        
        # Calculate statistics
        rate_stats = [
            np.min(rate_reqs),
            np.max(rate_reqs), 
            np.mean(rate_reqs),
            np.std(rate_reqs) if len(rate_reqs) > 1 else 0.0
        ]
        
        delay_stats = [
            np.min(delay_reqs),
            np.max(delay_reqs),
            np.mean(delay_reqs), 
            np.std(delay_reqs) if len(delay_reqs) > 1 else 0.0
        ]
        
        # Normalize using config ranges
        max_rate_req = self.config.get('max_rate_req', 10.0)
        max_delay_req = self.config.get('max_delay_req', 4.0)
        
        normalized_rate_stats = [stat / max_rate_req for stat in rate_stats]
        normalized_delay_stats = [stat / max_delay_req for stat in delay_stats]
        
        # Combine into collective info vector
        collective_info = np.array(normalized_rate_stats + normalized_delay_stats)
        
        # Ensure no NaN values
        collective_info = np.nan_to_num(collective_info, nan=0.0)
        
        return collective_info
    
    def _get_state(self):
        """
        Get the current state observation for the user with enhanced information.
        
        Returns:
            np.ndarray: Normalized state observation including collective user info.
        """        
        # Extract real location (x, y)
        real_location = self.user.real_location[:2]
        
        # Normalize location using the configured range
        x_min, x_max = self.x_y_location_range
        range_diff = x_max - x_min
        if range_diff > 0:
            normalized_location = (real_location - x_min) / range_diff
        else:
            normalized_location = np.array([0.5, 0.5])  # Default to center if range is invalid
            
        # Extract and normalize user requirements
        max_rate_req = self.config.get('max_rate_req', 10.0)  # Mbps
        max_delay_req = self.config.get('max_delay_req', 4.0)  # seconds
        
        rate_req = self.user.rate_requirement / max_rate_req if max_rate_req > 0 else 0.0
        delay_req = self.user.delay_requirement / max_delay_req if max_delay_req > 0 else 0.0
        
        # Calculate and normalize distances to all nodes (in real location)
        distances = []
        max_distance = np.sqrt((x_max - x_min)**2 * 2)  # Maximum possible distance in the area
        if max_distance == 0:
            max_distance = 1.0  # Prevent division by zero
            
        for node in self.nodes.values():
            distance = obj.calculate_distance(self.user, node, use_fake_distance=False)
            normalized_distance = distance / max_distance
            distances.append(min(normalized_distance, 1.0))  # Cap at 1.0
        
        # Include achievable throughput and delay if available (after resource allocation)
        if self.user.achieved_throughput is not None and not np.isnan(self.user.achieved_throughput):
            # Use reasonable defaults for normalization if not specified in config
            max_throughput = self.config.get('max_throughput', 100.0)  # Mbps
            max_delay = self.config.get('max_delay', 10.0)  # seconds
            
            # Safe division to avoid NaN
            if max_throughput > 0:
                norm_throughput = min(self.user.achieved_throughput / max_throughput, 1.0)
            else:
                norm_throughput = 0.0
                
            if max_delay > 0 and self.user.achieved_delay is not None and not np.isnan(self.user.achieved_delay):
                norm_delay = min(self.user.achieved_delay / max_delay, 1.0)
            else:
                norm_delay = 0.0
            
            # Cap values to prevent extreme outliers
            norm_throughput = np.clip(norm_throughput, 0.0, 1.0)
            norm_delay = np.clip(norm_delay, 0.0, 1.0)
        else:
            norm_throughput = 0.0
            norm_delay = 0.0
        
        # Include is_served as a binary feature
        is_served = 1.0 if self.user.is_served else 0.0
        
        # Get collective user information
        collective_info = self._get_collective_user_info()
        
        # Final safety check: Check for problematic values before they become NaN
        state_components = {
            'normalized_location': normalized_location,
            'rate_req': [rate_req],
            'delay_req': [delay_req],
            'distances': distances,
            'norm_throughput': [norm_throughput],
            'norm_delay': [norm_delay],
            'is_served': [is_served],
            'collective_info': collective_info
        }
        
        for name, component in state_components.items():
            component_array = np.array(component)
            if np.isnan(component_array).any():
                print(f"Warning: NaN detected in state component '{name}', replacing with zeros")
                component_array = np.nan_to_num(component_array, nan=0.0)
                state_components[name] = component_array
        
        # Combine all features into the normalized state vector
        state = np.concatenate([
            normalized_location,
            [rate_req],
            [delay_req],
            distances,
            [norm_throughput],
            [norm_delay],
            [is_served],
            collective_info
        ])
        
        # Check if any NaN or inf values exist and issue a warning
        if np.isnan(state).any():
            print(f"Warning: NaN values in final state for user {self.user.id}")
            state = np.nan_to_num(state, nan=0.0)
        
        if np.isinf(state).any():
            print(f"Warning: Inf values in final state for user {self.user.id}")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def step(self, action):
        """
        Execute the action in the environment.
        
        Args:
            action (np.ndarray): Action vector [noise_x, noise_y]
            where noise_x, noise_y are the sampled noise values to apply to the location.
            
        Returns:
            tuple: (None, 0, False, info)
                - next_state will be None, and should be retrieved after resource allocation
                - reward (float): Initial reward is 0, final reward after resource allocation.
                - done (bool): Whether the episode is done.
                - info (dict): Additional information.
        """
        # Use the sampled noise directly from the agent
        noise_x, noise_y = action[0], action[1]
        noise = np.array([noise_x, noise_y, 0])  # Append 0 for z-coordinate
        
        # Calculate epsilon based on the noise magnitude
        noise_magnitude = np.linalg.norm(noise[:2])
        if noise_magnitude > 0:
            epsilon = self.deltaF / noise_magnitude
        else:
            epsilon = float('inf')  # No privacy (no noise added)
        
        # Set epsilon for the user and generate fake location
        self.user.assigned_epsilon = epsilon
        
        # Directly set the fake location
        fake_location = self.user.real_location + noise
        
        # Ensure fake location is within valid range
        fake_location[:2] = np.clip(
            fake_location[:2], 
            self.x_y_location_range[0], 
            self.x_y_location_range[1]
        )
        
        self.user.fake_location = fake_location
        
        # Reset achievable metrics before new allocation
        self.user.achieved_throughput = None
        self.user.achieved_delay = None
        self.user.is_served = False
        
        # Record the action, noise, and resulting epsilon
        # (state will be updated after resource allocation)
        state_info = {
            'action': action,
            'sampled_noise': noise[:2],
            'epsilon': epsilon,
            'fake_location': fake_location,
            'is_served': False
        }
        self.history.append(state_info)
        
        # Return None for next_state as it will be calculated after resource allocation
        return None, 0, False, {'epsilon': epsilon, 'noise_magnitude': noise_magnitude}

    def get_next_state(self):
        """
        Get the next state after resource allocation has been completed.
        Should be called after resource allocation updates user metrics.
        
        Returns:
            np.ndarray: Next state observation.
        """
        next_state = self._get_state()
        
        # Update the state in the history
        if self.history:
            self.history[-1]['next_state'] = next_state
        
        return next_state
    
    def calculate_reward(self, is_served: bool):
        """
        Calculate the reward based on whether the user is served and the privacy level.
        
        Args:
            is_served (bool): Whether the user is served.
            
        Returns:
            float: Calculated reward.
        """
        # Get the latest history entry
        if not self.history:
            return 0.0
        
        latest = self.history[-1]
        epsilon = latest['epsilon']
        noise_magnitude = np.linalg.norm(latest['sampled_noise'])
        
        # Update the is_served status in the history
        latest['is_served'] = is_served
        
        # Calculate base reward using the noise magnitude
        base_reward = noise_magnitude
        
        # Multiply by service indicator: 1 if served, -1 if not served
        service_factor = 1 if is_served else -1
        reward = service_factor * base_reward if is_served else -1
        
        # Special case for no privacy (epsilon = infinity)
        if epsilon == float('inf') and is_served:
            reward = -10  # Heavy penalty for no privacy
        
        return reward    
    
    def update_reward(self, is_served: bool):
        """
        Update the latest history entry with the calculated reward and return the reward.
        For getting the next state, call get_next_state() after this method.
        
        Args:
            is_served (bool): Whether the user is served.
            
        Returns:
            float: Calculated reward.
        """
        reward = self.calculate_reward(is_served)
        
        if self.history:
            self.history[-1]['reward'] = reward
            
        return reward
    
    def get_next_state_continuous(self):
        """
        Get the next state for continuous infinite horizon training.
        This method transitions to the next state without resetting.
        
        Returns:
            np.ndarray: Next state observation.
        """
        # For infinite horizon, we don't reset - just get current state
        return self._get_state()
    
    def transition_to_next_iteration(self):
        """
        Transition the environment to the next iteration for infinite horizon training.
        This updates the user's real location slightly to simulate movement over time.
        """
        # Introduce small random movement to simulate user mobility
        movement_std = self.config.get('user_movement_std', 0.01)  # Small movement
        
        # Add small random movement to real location
        movement = np.random.normal(0, movement_std, 2)  # Only x, y movement
        new_location = self.user.real_location[:2] + movement
        
        # Ensure location stays within bounds
        x_min, x_max = self.x_y_location_range
        new_location = np.clip(new_location, x_min, x_max)
        
        # Update real location
        self.user.real_location[:2] = new_location
        
        # Reset allocation results for next iteration
        self.user.is_served = False
        self.user.associated_BS = None
        self.user.associated_server = None
        self.user.achieved_throughput = None
        self.user.achieved_delay = None
