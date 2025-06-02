# Infinite Horizon Training Analysis

## Current Implementation (Episodic)
The current system follows this pattern:
1. **Episode Loop**: Multiple episodes (num_episodes)
2. **Step Loop**: Multiple steps per episode (steps_per_episode) 
3. **Environment Reset**: Between episodes, environments are reset
4. **Done Flags**: Episodes terminate with done=True

## Required Implementation (Infinite Horizon)
For infinite horizon, on-policy, single-step training:
1. **Single Iteration Loop**: One continuous loop for a fixed number of iterations
2. **No Episodes**: No concept of episode boundaries or resets
3. **Continuous State Transitions**: Each iteration transitions from current state to next state
4. **No Done Flags**: No termination conditions except reaching max iterations
5. **Single-Step Updates**: Update policy after each single transition

## Key Differences

### Current (Episodic):
```python
for episode in range(num_episodes):
    states = {user_id: env.reset() for user_id, env in self.user_envs.items()}  # RESET!
    for step in range(steps_per_episode):
        # Take actions, get rewards, collect experience
        # Mark done=True at end of episode
    # Update policy with batch of episode data
```

### Required (Infinite Horizon):
```python
# Initialize states once at the beginning
states = {user_id: env.reset() for user_id, env in self.user_envs.items()}

for iteration in range(max_iterations):
    # Take actions from current states
    # Get rewards and next states
    # Update policy immediately (on-policy)
    # Continue from next states (NO RESET)
    states = next_states  # Continue seamlessly
```

## Implications for the System

1. **No Environment Resets**: Environments should not reset between iterations
2. **No Done Flags**: All done flags should be False (or eliminated entirely)
3. **GAE Simplification**: With single-step transitions, GAE becomes trivial
4. **Memory Management**: Only need to store single transitions, not episode batches
5. **Discount Factor**: For single-step updates, gamma becomes irrelevant

## Recommended Changes

1. **Restructure Training Loop**: Remove episodic structure
2. **Modify Environment**: Remove reset behavior between iterations  
3. **Update GAE Computation**: Simplify for single-step case
4. **Revise Memory**: Store single transitions instead of episode batches
