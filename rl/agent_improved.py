import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from .nan_diagnostics import diagnose_tensor_issues, fix_division_by_zero

def validate_tensor(tensor, name="tensor", allow_nan=False):
    """Validates a tensor and raises an error if NaN/Inf is found when not allowed."""
    if tensor is None:
        return tensor
    
    if torch.isnan(tensor).any():
        if not allow_nan:
            print(f"Warning: NaN values detected in {name}")
    
    if torch.isinf(tensor).any():
        diagnose_tensor_issues(tensor, name)
        raise ValueError(f"Inf detected in {name}. Training should stop to investigate root cause.")
    
    return tensor

def safe_division(numerator, denominator, min_denominator=1e-8):
    """Performs safe division avoiding division by zero."""
    return fix_division_by_zero(numerator, denominator, epsilon=min_denominator)

class PPOActor(nn.Module):
    """
    Actor network for the PPO algorithm. It outputs the parameters of a
    distribution (mean and standard deviation) for the location perturbation.
    """
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initialize the actor network.
        
        Args:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(PPOActor, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output means for 2 independent actions: noise_x, noise_y
        self.mean_head = nn.Linear(hidden_dim, 2)
        # Output log standard deviations for 2 independent actions
        self.log_std_head = nn.Linear(hidden_dim, 2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights of the network"""
        for layer in [self.fc1, self.fc2, self.mean_head]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
        
        # Initialize std with small values for stable training
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # Start with small std (exp(-1) ≈ 0.37)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor containing the features.
            
        Returns:
            tuple: (mean, std)
                - mean (torch.Tensor): Mean for 2 actions [noise_x, noise_y], constrained to [-1, 1].
                - std (torch.Tensor): Standard deviation for 2 actions, constrained to reasonable range.
        """
        # Validate input tensor
        x = validate_tensor(x, "actor_input")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Apply tanh to constrain means to [-1, 1]
        mean = torch.tanh(self.mean_head(x))
        
        # Use log std for numerical stability, then exp to get std
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent extreme values
        std = torch.exp(log_std)
        
        # Validate outputs
        validate_tensor(mean, "actor_mean")
        validate_tensor(std, "actor_std")
        return mean, std
        
    def get_action(self, state, deterministic=False, deltaF=None):
        """
        Sample actions from the policy distribution.
        
        Args:
            state (torch.Tensor): Input state tensor.
            deterministic (bool): If True, return mean instead of sampling.
            deltaF (float): Privacy parameter to scale the actions.
            
        Returns:
            tuple: (action, log_prob)
                - action (torch.Tensor): Sampled actions [noise_x, noise_y].
                - log_prob (torch.Tensor): Log probability of the sampled actions.
        """
        # Validate input state
        state = validate_tensor(state, "get_action_state")
            
        mean, std = self.forward(state)
        
        # Create Normal distributions for each action dimension
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        
        # Calculate log probability of the action
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions
        
        # Scale by deltaF if provided (scale the action, not the distribution parameters)
        if deltaF is not None:
            action = action * deltaF
        
        # Validate outputs
        action = validate_tensor(action, "sampled_action")
        log_prob = validate_tensor(log_prob, "action_log_prob")
        return action, log_prob
        
    def evaluate_actions(self, state, action):
        """
        Evaluate the log probability and entropy of given actions.
        
        Standard PPO evaluation for continuous actions.
        
        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Actions to evaluate [noise_x, noise_y].
            
        Returns:
            tuple: (log_prob, entropy)
                - log_prob (torch.Tensor): Log probability of the actions.
                - entropy (torch.Tensor): Entropy of the current policy distribution.
        """
        # Validate inputs
        state = validate_tensor(state, "evaluate_state")
        action = validate_tensor(action, "evaluate_action")
        
        # Get current policy parameters
        mean, std = self.forward(state)
        
        # Create distribution with current policy parameters
        dist = Normal(mean, std)
        
        # Calculate log probability of the given actions
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions
        
        # Calculate entropy of the current policy
        entropy = dist.entropy().sum(-1)  # Sum over action dimensions
        
        # Validate outputs
        validate_tensor(log_prob, "evaluate_log_prob")
        validate_tensor(entropy, "evaluate_entropy")
        
        return log_prob, entropy


class PPOCritic(nn.Module):
    """
    Critic network for the PPO algorithm. It estimates the value function.
    """
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initialize the critic network.
        
        Args:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(PPOCritic, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the network"""
        for layer in [self.fc1, self.fc2, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor containing the features.
            
        Returns:
            torch.Tensor: Estimated state value.
        """
        # Validate input
        x = validate_tensor(x, "critic_input")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        
        # Validate output
        validate_tensor(value, "critic_value")
        
        return value


class PPOAgent:
    """
    PPO Agent that combines actor and critic networks.
    Enhanced for infinite-horizon training with multi-epoch updates and multi-step returns.
    """
    def __init__(
        self,
        state_dim,
        device='cpu',
        hidden_dim=128,
        lr_actor=3e-4,
        lr_critic=3e-4,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        multi_step_returns=3,
        gamma=0.99
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space.
            device (str): Device to run the model on ('cpu' or 'cuda').
            hidden_dim (int): Dimension of hidden layers.
            lr_actor (float): Learning rate for the actor.
            lr_critic (float): Learning rate for the critic.
            clip_param (float): PPO clipping parameter.
            value_loss_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy coefficient.
            max_grad_norm (float): Maximum norm of the gradients.
            ppo_epochs (int): Number of PPO update epochs per batch.
            multi_step_returns (int): Number of steps for multi-step returns.
            gamma (float): Discount factor.
        """
        self.device = torch.device(device)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.multi_step_returns = multi_step_returns
        self.gamma = gamma
        
        # Create actor and critic networks
        self.actor = PPOActor(state_dim, hidden_dim).to(self.device)
        self.critic = PPOCritic(state_dim, hidden_dim).to(self.device)
        
        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def get_action(self, state, deterministic=False, deltaF=None):
        """
        Sample an action from the policy.
        
        Args:
            state (np.ndarray or torch.Tensor): State observation.
            deterministic (bool): If True, return the mean of the distribution.
            deltaF (float): Privacy parameter to scale the sampled actions.
            
        Returns:
            tuple: (action, log_prob)
                - action (np.ndarray): Sampled actions [noise_x, noise_y].
                - log_prob (float): Log probability of the sampled actions.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic, deltaF)
        
        return action.cpu().numpy().squeeze(), log_prob.cpu().numpy().squeeze()
    
    def compute_multi_step_returns(self, rewards, values, next_values, dones, gamma=None):
        """
        Compute multi-step returns using TD(n).
        
        Args:
            rewards (torch.Tensor): Rewards for each step.
            values (torch.Tensor): Current state values.
            next_values (torch.Tensor): Next state values.
            dones (torch.Tensor): Done flags.
            gamma (float): Discount factor.
            
        Returns:
            torch.Tensor: Multi-step returns.
        """
        if gamma is None:
            gamma = self.gamma
            
        returns = torch.zeros_like(rewards)
        
        for t in range(len(rewards)):
            return_t = 0
            for k in range(self.multi_step_returns):
                if t + k < len(rewards):
                    if t + k < len(dones) and dones[t + k]:
                        # Episode ended
                        return_t += (gamma ** k) * rewards[t + k]
                        break
                    else:
                        return_t += (gamma ** k) * rewards[t + k]
                else:
                    # Use next state value for bootstrapping
                    if t + k < len(next_values):
                        return_t += (gamma ** k) * next_values[t + k]
                    break
            
            # Add bootstrapped value if we didn't hit a terminal state
            if t + self.multi_step_returns < len(next_values) and not (t + self.multi_step_returns < len(dones) and dones[t + self.multi_step_returns]):
                return_t += (gamma ** self.multi_step_returns) * next_values[t + self.multi_step_returns]
            
            returns[t] = return_t
        
        return returns
    
    def update_batch(self, states, actions, log_probs_old, rewards, next_states, dones):
        """
        Update the actor and critic networks using batch PPO with multiple epochs.
        
        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            log_probs_old (torch.Tensor): Batch of log probabilities from the old policy.
            rewards (torch.Tensor): Batch of rewards.
            next_states (torch.Tensor): Batch of next states.
            dones (torch.Tensor): Batch of done flags.
            
        Returns:
            dict: Dictionary containing the loss metrics.
        """
        # Move data to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Validate inputs
        validate_tensor(states, "update_states")
        validate_tensor(actions, "update_actions") 
        validate_tensor(log_probs_old, "update_log_probs_old")
        validate_tensor(rewards, "update_rewards")
        validate_tensor(next_states, "update_next_states")
        
        # Compute values for current and next states
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
        
        # Compute multi-step returns
        returns = self.compute_multi_step_returns(rewards, values, next_values, dones)
        
        # Compute advantages
        advantages = returns - values
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Multiple epochs of PPO updates
        for epoch in range(self.ppo_epochs):
            # Evaluate actions with current policy
            log_probs_new, entropy = self.actor.evaluate_actions(states, actions)
            
            # Calculate importance sampling ratio
            ratio = torch.exp(log_probs_new - log_probs_old)
            
            # Calculate actor loss (PPO clipped objective)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
              # Calculate critic loss - Standard PPO approach
            current_values = self.critic(states).squeeze(-1)
            
            # Standard PPO critic loss: learn to predict returns accurately
            critic_loss = F.mse_loss(current_values, returns)
            
            # Calculate entropy loss
            entropy_loss = -entropy.mean()
            
            # Total losses
            total_loss_actor = actor_loss + self.entropy_coef * entropy_loss
            total_loss_critic = self.value_loss_coef * critic_loss
            
            # Update actor
            self.actor_optimizer.zero_grad()
            total_loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            total_loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # Accumulate losses
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        return {
            'actor_loss': total_actor_loss / self.ppo_epochs,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs
        }

    def get_state_dict(self):
        """
        Get the state dictionaries of the actor and critic networks.
        
        Returns:
            dict: Dictionary containing the state dictionaries.
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        
    def set_state_dict(self, state_dict):
        """
        Set the state dictionaries of the actor and critic networks.
        
        Args:
            state_dict (dict): Dictionary containing the state dictionaries.
        """
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        if 'actor_optimizer' in state_dict:
            self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        if 'critic_optimizer' in state_dict:
            self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
