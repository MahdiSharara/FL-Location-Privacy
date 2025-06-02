import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple
from collections import defaultdict
from .agent_improved import PPOAgent
from .environment_improved import UserEnvImproved
from .federated_learning import FederatedLearner
from .analyzer import RLAnalyzer

class InfiniteHorizonTrainer:
    """
    Improved trainer for infinite-horizon RL with enhanced features:
    - Multi-step returns
    - Multiple PPO epochs per batch
    - Enhanced state representation with collective user info
    - On-policy batch updates
    """
    
    def __init__(self, config: dict, users: Dict, nodes: Dict, links: Dict, 
                 resource_allocator, device='cpu'):
        """
        Initialize the infinite horizon trainer.
        
        Args:
            config (dict): Configuration dictionary.
            users (Dict): Dictionary of users.
            nodes (Dict): Dictionary of nodes.
            links (Dict): Dictionary of links.
            resource_allocator: Resource allocation algorithm.
            device (str): Device to run training on.
        """
        self.config = config
        self.users = users
        self.nodes = nodes
        self.links = links
        self.resource_allocator = resource_allocator
        self.device = device
        
        # Training parameters
        self.batch_size = config.get('batch_size', 64)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.multi_step_returns = config.get('multi_step_returns', 3)
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate_actor = config.get('lr_actor', 3e-4)
        self.learning_rate_critic = config.get('lr_critic', 3e-4)
        
        # Create environments for each user with collective information
        self.envs = {}
        for user_id, user in users.items():
            self.envs[user_id] = UserEnvImproved(
                user=user,
                nodes=nodes,
                links=links,
                all_users=users,  # Pass all users for collective info
                config=config
            )
        
        # Get state dimension from first environment (all should be the same)
        first_env = next(iter(self.envs.values()))
        state_dim = first_env.state_dim
        
        # Create agents for each user
        self.agents = {}
        for user_id in users.keys():
            self.agents[user_id] = PPOAgent(
                state_dim=state_dim,
                device=device,
                lr_actor=self.learning_rate_actor,
                lr_critic=self.learning_rate_critic,
                ppo_epochs=self.ppo_epochs,
                multi_step_returns=self.multi_step_returns,
                gamma=self.gamma
            )
        
        # Federated learning setup
        self.federated_learner = None
        if config.get('federated_learning', False):
            self.federated_learner = FederatedLearner(
                num_users=len(users),
                federate_actor=config.get('federate_actor', True),
                federate_critic=config.get('federate_critic', True),
                dp_eps=config.get('dp_eps', None),
                dp_clip=config.get('dp_clip', 1.0)
            )
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.analyzer = RLAnalyzer()
        
        # Batch collection for on-policy training
        self.reset_batch_data()
        
    def reset_batch_data(self):
        """Reset batch data collection."""
        self.batch_data = {
            user_id: {
                'states': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'next_states': [],
                'dones': []
            } for user_id in self.users.keys()
        }
    
    def collect_batch_data(self, iteration):
        """
        Collect batch data for on-policy training.
        
        Args:
            iteration (int): Current iteration number.
            
        Returns:
            dict: Collected batch data statistics.
        """
        batch_stats = {
            'total_served': 0,
            'total_users': len(self.users),
            'avg_epsilon': 0.0,
            'avg_reward': 0.0,
            'total_reward': 0.0
        }
        
        # Collect current states for all users
        current_states = {}
        for user_id, env in self.envs.items():
            current_states[user_id] = env._get_state()
        
        # Generate actions for all users
        actions = {}
        log_probs = {}
        for user_id, agent in self.agents.items():
            state = current_states[user_id]
            action, log_prob = agent.get_action(
                state, 
                deterministic=False,
                deltaF=self.config.get('deltaF', 1.0)
            )
            actions[user_id] = action
            log_probs[user_id] = log_prob
        
        # Execute actions in environments
        for user_id, env in self.envs.items():
            env.step(actions[user_id])
        
        # Perform resource allocation
        allocation_results = self.resource_allocator.allocate()
        
        # Get next states and rewards
        next_states = {}
        rewards = {}
        total_epsilon = 0.0
        served_count = 0
        
        for user_id, env in self.envs.items():
            user = self.users[user_id]
            is_served = user.is_served
            
            # Calculate reward and get next state
            reward = env.update_reward(is_served)
            next_state = env.get_next_state()
            
            # Store data
            next_states[user_id] = next_state
            rewards[user_id] = reward
            
            # Update statistics
            if is_served:
                served_count += 1
            
            if hasattr(user, 'assigned_epsilon') and user.assigned_epsilon != float('inf'):
                total_epsilon += user.assigned_epsilon
            
            batch_stats['total_reward'] += reward
            
            # Add to batch data
            self.batch_data[user_id]['states'].append(current_states[user_id])
            self.batch_data[user_id]['actions'].append(actions[user_id])
            self.batch_data[user_id]['log_probs'].append(log_probs[user_id])
            self.batch_data[user_id]['rewards'].append(reward)
            self.batch_data[user_id]['next_states'].append(next_state)
            self.batch_data[user_id]['dones'].append(False)  # Infinite horizon
        
        # Update batch statistics
        batch_stats['total_served'] = served_count
        batch_stats['avg_epsilon'] = total_epsilon / len(self.users) if len(self.users) > 0 else 0.0
        batch_stats['avg_reward'] = batch_stats['total_reward'] / len(self.users) if len(self.users) > 0 else 0.0
        
        # Transition environments to next iteration
        for env in self.envs.values():
            env.transition_to_next_iteration()
        
        return batch_stats
    
    def update_agents_batch(self):
        """
        Update all agents using collected batch data with multiple epochs.
        
        Returns:
            dict: Training statistics.
        """
        update_stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0, 
            'entropy': 0.0
        }
        
        num_updates = 0
        
        for user_id, agent in self.agents.items():
            user_batch = self.batch_data[user_id]
            
            # Check if we have enough data
            if len(user_batch['states']) < 2:
                continue
            
            # Convert to numpy arrays
            states = np.array(user_batch['states'])
            actions = np.array(user_batch['actions'])
            log_probs = np.array(user_batch['log_probs'])
            rewards = np.array(user_batch['rewards'])
            next_states = np.array(user_batch['next_states'])
            dones = np.array(user_batch['dones'])
            
            # Update agent
            loss_info = agent.update_batch(
                states, actions, log_probs, rewards, next_states, dones
            )
            
            # Accumulate statistics
            update_stats['actor_loss'] += loss_info['actor_loss']
            update_stats['critic_loss'] += loss_info['critic_loss']
            update_stats['entropy'] += loss_info['entropy']
            num_updates += 1
        
        # Average the statistics
        if num_updates > 0:
            for key in update_stats:
                update_stats[key] /= num_updates
        
        # Federated learning update
        if self.federated_learner is not None:
            aggregated_state = self.federated_learner.aggregate(self.agents)
            if aggregated_state is not None:
                self.federated_learner.apply_updates(self.agents, aggregated_state)
        
        return update_stats
    
    def train_infinite_horizon(self, total_iterations: int, 
                             update_frequency: int = 32,
                             save_frequency: int = 1000,
                             eval_frequency: int = 500,
                             save_path: str = None):
        """
        Train agents using infinite horizon with enhanced features.
        
        Args:
            total_iterations (int): Total number of training iterations.
            update_frequency (int): How often to update agents (in iterations).
            save_frequency (int): How often to save models.
            eval_frequency (int): How often to evaluate models.
            save_path (str): Path to save models and results.
        """
        logging.info(f"Starting infinite horizon training for {total_iterations} iterations")
        
        start_time = time.time()
        
        for iteration in range(total_iterations):
            # Collect batch data
            batch_stats = self.collect_batch_data(iteration)
            
            # Update agents every update_frequency iterations
            if (iteration + 1) % update_frequency == 0:
                update_stats = self.update_agents_batch()
                
                # Reset batch data after update
                self.reset_batch_data()
                
                # Log training progress
                logging.info(f"Iteration {iteration + 1}/{total_iterations}")
                logging.info(f"  Served: {batch_stats['total_served']}/{batch_stats['total_users']}")
                logging.info(f"  Avg Reward: {batch_stats['avg_reward']:.4f}")
                logging.info(f"  Avg Epsilon: {batch_stats['avg_epsilon']:.4f}")
                logging.info(f"  Actor Loss: {update_stats['actor_loss']:.4f}")
                logging.info(f"  Critic Loss: {update_stats['critic_loss']:.4f}")
                logging.info(f"  Entropy: {update_stats['entropy']:.4f}")
                
                # Store metrics
                self.metrics['iteration'].append(iteration + 1)
                self.metrics['served_percentage'].append(
                    batch_stats['total_served'] / batch_stats['total_users'] * 100
                )
                self.metrics['avg_reward'].append(batch_stats['avg_reward'])
                self.metrics['avg_epsilon'].append(batch_stats['avg_epsilon'])
                self.metrics['actor_loss'].append(update_stats['actor_loss'])
                self.metrics['critic_loss'].append(update_stats['critic_loss'])
                self.metrics['entropy'].append(update_stats['entropy'])
            
            # Save models periodically
            if save_path and (iteration + 1) % save_frequency == 0:
                self.save_models(f"{save_path}/models_iter_{iteration + 1}")
                self.save_metrics(f"{save_path}/metrics_iter_{iteration + 1}.json")
            
            # Evaluation
            if (iteration + 1) % eval_frequency == 0:
                eval_stats = self.evaluate_performance()
                logging.info(f"Evaluation at iteration {iteration + 1}: {eval_stats}")
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final save
        if save_path:
            self.save_models(f"{save_path}/final_models")
            self.save_metrics(f"{save_path}/final_metrics.json")
        
        return self.metrics
    
    def evaluate_performance(self, num_eval_episodes: int = 100):
        """
        Evaluate the current policy performance.
        
        Args:
            num_eval_episodes (int): Number of episodes for evaluation.
            
        Returns:
            dict: Evaluation statistics.
        """
        eval_stats = {
            'avg_reward': 0.0,
            'served_percentage': 0.0,
            'avg_epsilon': 0.0
        }
        
        total_reward = 0.0
        total_served = 0
        total_epsilon = 0.0
        
        for episode in range(num_eval_episodes):
            # Reset environments
            for env in self.envs.values():
                env.reset()
            
            # Generate actions deterministically
            for user_id, agent in self.agents.items():
                state = self.envs[user_id]._get_state()
                action, _ = agent.get_action(state, deterministic=True)
                self.envs[user_id].step(action)
            
            # Perform allocation
            self.resource_allocator.allocate()
            
            # Collect statistics
            for user_id, env in self.envs.items():
                user = self.users[user_id]
                reward = env.calculate_reward(user.is_served)
                total_reward += reward
                
                if user.is_served:
                    total_served += 1
                
                if hasattr(user, 'assigned_epsilon') and user.assigned_epsilon != float('inf'):
                    total_epsilon += user.assigned_epsilon
        
        # Calculate averages
        total_evaluations = num_eval_episodes * len(self.users)
        eval_stats['avg_reward'] = total_reward / total_evaluations
        eval_stats['served_percentage'] = (total_served / total_evaluations) * 100
        eval_stats['avg_epsilon'] = total_epsilon / total_evaluations
        
        return eval_stats
    
    def save_models(self, path: str):
        """Save all agent models."""
        import os
        os.makedirs(path, exist_ok=True)
        
        for user_id, agent in self.agents.items():
            torch.save(agent.get_state_dict(), f"{path}/agent_{user_id}.pt")
        
        if self.federated_learner is not None:
            torch.save(self.federated_learner.get_state(), f"{path}/federated_learner.pt")
    
    def load_models(self, path: str):
        """Load all agent models."""
        for user_id, agent in self.agents.items():
            model_path = f"{path}/agent_{user_id}.pt"
            if os.path.exists(model_path):
                agent.set_state_dict(torch.load(model_path, map_location=self.device))
        
        if self.federated_learner is not None:
            fl_path = f"{path}/federated_learner.pt"
            if os.path.exists(fl_path):
                self.federated_learner.set_state(torch.load(fl_path, map_location=self.device))
    
    def save_metrics(self, path: str):
        """Save training metrics."""
        import json
        with open(path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = {}
            for key, values in self.metrics.items():
                serializable_metrics[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            json.dump(serializable_metrics, f, indent=2)
    
    def get_training_summary(self):
        """
        Get a summary of the training process.
        
        Returns:
            dict: Training summary statistics.
        """
        if not self.metrics['avg_reward']:
            return {"status": "No training data available"}
        
        return {
            "total_iterations": len(self.metrics['avg_reward']),
            "final_avg_reward": self.metrics['avg_reward'][-1],
            "final_served_percentage": self.metrics['served_percentage'][-1],
            "final_avg_epsilon": self.metrics['avg_epsilon'][-1],
            "best_avg_reward": max(self.metrics['avg_reward']),
            "best_served_percentage": max(self.metrics['served_percentage']),
            "avg_actor_loss": np.mean(self.metrics['actor_loss']),
            "avg_critic_loss": np.mean(self.metrics['critic_loss']),
            "avg_entropy": np.mean(self.metrics['entropy'])
        }
