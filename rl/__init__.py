"""
Reinforcement Learning components for Federated Learning with Location Privacy.

This package contains RL-based implementations for:
- Training agents for federated learning scenarios
- Resource allocation optimization
- Environment modeling
- Performance analysis
"""

# Import main components to make them available at package level
try:
    from .trainer_improved import InfiniteHorizonTrainer
    from .resource_allocator import ResourceAllocator
    from .environment_improved import UserEnvImproved
    from .agent_improved import PPOAgent
    from .analyzer import RLAnalyzer
    from .federated_learning import FederatedLearner
    
    __all__ = [
        'InfiniteHorizonTrainer',
        'ResourceAllocator', 
        'UserEnvImproved',
        'PPOAgent',
        'RLAnalyzer',
        'FederatedLearner'
    ]
except ImportError as e:
    # In case of missing dependencies, provide a graceful fallback
    import warnings
    warnings.warn(f"Some RL components could not be imported: {e}")
    __all__ = []
