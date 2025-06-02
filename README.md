# Reinforcement Learning for Location Privacy in Federated Learning

This project implements a reinforcement learning (RL) approach to location privacy in federated learning networks. The RL agents learn to generate optimal noise for location obfuscation while ensuring quality of service (QoS) requirements in mobile communication systems.

## 🚀 Features

- **PPO-based RL agents** for intelligent location noise generation
- **Federated Learning integration** with three modes: both, critic-only, or none
- **Comprehensive simulation framework** for mobile network resource allocation
- **Interactive visualization** with Plotly and Dash
- **Infinite horizon training** for improved learning convergence
- **Privacy-utility trade-off optimization**

## Overview

In mobile networks, users often need to share their locations to receive services, but this raises privacy concerns. This project develops an RL-based privacy solution where each user is represented by an agent that learns to add optimal noise to the user's real location. The fake locations are then used by the network to allocate resources. Our approach balances privacy preservation with maintaining acceptable quality of service.

## Architecture

### Core RL Components

1. **Agent (`rl/agent.py`)**: 
   - PPO-based actor-critic architecture
   - Actor network that outputs location noise parameters (mean and standard deviation)
   - Critic network that estimates state values
   - Policy optimization using clipped surrogate objective

2. **Environment (`rl/environment.py`)**: 
   - UserEnv class representing each user's environment
   - State space includes user location, requirements, and distances to nodes
   - Action space is the noise to apply to locations
   - Reward function balances privacy (higher epsilon) with service quality

3. **Resource Allocator (`rl/resource_allocator.py`)**:
   - Heuristic algorithm for user-BS association and resource block allocation
   - Network routing using shortest path algorithms
   - Optimization of transmission power and modulation selection
   - Server selection based on processing capacity and link quality
   
4. **Federated Learning (`rl/federated_learning.py`)**:
   - Model aggregation framework for sharing learning across agents
   - Support for three federated modes: both (actor+critic), critic-only, or none
   - Optional differential privacy to protect individual agent contributions
   - Weighted aggregation based on local data contribution

5. **Trainer (`rl/trainer.py`)**:
   - Orchestrates the entire training process across all agents
   - Manages experience collection and policy updates
   - Implements federated learning rounds between training episodes
   - Handles evaluation and metrics tracking

6. **Analyzer (`rl/analyzer.py`)**:
   - Tracks and processes training metrics
   - Visualizes learning curves and convergence statistics
   - Compares different federated learning configurations
   - Generates comprehensive training reports

### Integration With Simulation Framework

The RL solution is integrated with the existing simulation framework through:

- **Simulation module (`simulation.py`)**:
  - Added `run_rl_solution` function to handle RL-based simulations
  - Maintains consistent metrics with previous implementations
  
- **Dedicated RL scenarios script (`run_rl_scenarios.py`)**:
  - Streamlined interface for running RL experiments
  - Scenario configuration and batch execution
  - Results collection and analysis

## Training Configurations

The framework supports multiple federated learning configurations:

1. **Full Federated Learning (`both`)**: Both the actor and critic models are trained using federated learning, maximizing knowledge sharing across users.
2. **Critic-only Federated Learning (`critic`)**: Only the value function (critic network) uses federated learning; the actor policy is trained locally. This balances personalization with global value estimation.
3. **No Federated Learning (`none`)**: Each agent trains completely independently, with models optimized for individual circumstances.

Each configuration can also use differential privacy (DP) to protect the privacy of model updates during federated learning.

## Usage Instructions

### Running RL Scenarios

To run predefined RL scenarios:

```bash
python run_rl_scenarios.py
```

This script will:
1. Train RL agents with various federated learning configurations
2. Evaluate the trained policies on privacy and QoS metrics
3. Generate visualization of results in the `results` directory

### Performance Benchmarking

For comprehensive performance evaluation across different network configurations:

```bash
python benchmark_rl_performance.py --users 10 20 30 --nodes 3 5 --modes both critic none
```

Command line arguments:
- `--users`: List of user counts to test
- `--nodes`: List of node counts to test
- `--modes`: Federated learning modes to evaluate
- `--dp`: Differential privacy epsilon values (use -1 for None)
- `--runs`: Number of runs per configuration
- `--episodes`: Number of training episodes
- `--batch-size`: Batch size for training
- `--no-parallel`: Disable parallel execution

### Integration Testing

To verify the core RL functionality:

```bash
python test_rl_integration.py
```

This runs a simplified validation of all RL components and generates quick comparison plots.

## Configuration Options

The main parameters can be configured in `config.json`:

```json
{
  "x_y_location_range": [-2000.0, 2000.0],  // Area boundaries in meters
  "deltaF": 4000,                           // Distance factor for privacy calculations
  "n_RBs": 20,                              // Number of resource blocks
  "n_MCS": 14,                              // Number of modulation and coding schemes
  "delay_requirement": [0.5, 2],            // Delay requirements range (seconds per MB)
  "rate_requirement": [5, 10],              // Rate requirements range (Mbps)
  "P_max_Tx_dBm": 23.0                      // Maximum transmission power (dBm)
}
```

RL-specific parameters can be adjusted in the scenario definitions:

```python
# Example scenario configuration
scenario = {
    "name": "RL-Full-Federated-Learning",
    "use_rl": True,
    "federate_mode": "both",        # Options: "both", "critic", "none"
    "differential_privacy": 0.5,     # None for no DP, or epsilon value
    "episodes": 300,                # Number of training episodes
    "batch_size": 64,               # Batch size for PPO updates
    "max_noise_radius": 1000        # Maximum noise distance in meters
}
```

## Key Components Explained

### PPO Agent Architecture

The PPO (Proximal Policy Optimization) agent uses a dual-network architecture:

- **Actor Network**: Outputs the mean and standard deviation of a Gaussian distribution for sampling location noise. This stochastic policy enables exploration while training.

- **Critic Network**: Estimates the value function (expected future rewards) to help stabilize training and reduce variance.

- **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE) to improve sample efficiency and training stability.

### Reward Function

The reward function balances privacy with quality of service:

```python
reward = (privacy_weight * epsilon) + (qos_weight * is_served) - (delay_penalty * normalized_delay)
```

Where:
- `epsilon` is the privacy measure (higher is better)
- `is_served` is a binary indicator of service quality
- `normalized_delay` penalizes excessive delays

### Federated Learning Process

The federated learning process follows these steps:
1. Local training of agents using PPO
2. Model parameter collection from all agents
3. Weighted aggregation of parameters (optionally with differential privacy)
4. Distribution of aggregated model to all agents
5. Continuation of training with the updated models

## Results Analysis

The analyzer module provides comprehensive analysis of training results:

- Training curves showing reward, epsilon, and service rate evolution
- Convergence statistics to measure learning stability
- Comparison between different federated learning configurations
- Privacy-QoS tradeoff visualization
- Per-user performance breakdown

## Conclusion

This reinforcement learning approach to location privacy demonstrates that:

1. RL agents can learn to add optimal noise that preserves privacy while maintaining QoS
2. Federated learning improves convergence speed and stability
3. Different FL configurations offer various tradeoffs between personalization and collective learning
4. The solution adapts to different network conditions and user requirements

Future work could explore:
- More complex network topologies
- Adversarial training to improve robustness
- Integration with other privacy-preserving techniques
- Dynamic adaptation to changing user preferences

- `fl_mode`: Federation mode (`"both"`, `"actor"`, `"critic"`, or `"none"`)
- `training_episodes`: Number of training episodes
- `steps_per_episode`: Number of steps per episode
- `fl_rounds`: Frequency of federated aggregation
- `cancel_privacy`: Whether to disable privacy (for baseline)

## Monitoring Training

The RL framework provides several metrics for monitoring training:

- **Average Reward**: Overall performance of the agents
- **Served Percentage**: Percentage of users successfully served
- **Average Epsilon**: Privacy level achieved
- **Actor/Critic Loss**: Network training losses
- **Entropy**: Policy exploration metric

## Integrating with Existing Code

The RL framework has been designed to work alongside the existing MILP-based solution. Both approaches share:

- Data structures (User, Node, Link)
- Network generation
- Channel gain calculations
- Results analysis and plotting

## Requirements

The RL components require PyTorch and NetworkX in addition to the existing dependencies.
