"""
NaN and numerical instability diagnostics for PyTorch tensors.

This module provides utilities to diagnose and fix numerical issues
that can occur during neural network training, particularly with RL algorithms.
"""

import torch
import numpy as np
import logging


def diagnose_tensor_issues(tensor, name="tensor"):
    """
    Diagnose numerical issues in a tensor and log detailed information.
    
    Args:
        tensor (torch.Tensor): The tensor to diagnose
        name (str): Name of the tensor for logging purposes
    """
    if tensor is None:
        logging.warning(f"Tensor '{name}' is None")
        return
    
    if not isinstance(tensor, torch.Tensor):
        logging.warning(f"'{name}' is not a torch.Tensor (type: {type(tensor)})")
        return
    
    # Basic tensor info
    logging.info(f"Tensor '{name}' diagnostics:")
    logging.info(f"  Shape: {tensor.shape}")
    logging.info(f"  Device: {tensor.device}")
    logging.info(f"  Dtype: {tensor.dtype}")
    logging.info(f"  Requires grad: {tensor.requires_grad}")
    
    # Check for numerical issues
    if tensor.numel() == 0:
        logging.warning(f"  Empty tensor")
        return
    
    # Convert to float for analysis if needed
    if tensor.dtype in [torch.int32, torch.int64, torch.bool]:
        analysis_tensor = tensor.float()
    else:
        analysis_tensor = tensor
    
    # NaN detection
    nan_count = torch.isnan(analysis_tensor).sum().item()
    if nan_count > 0:
        logging.error(f"  NaN values: {nan_count}/{tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)")
        
        # Find NaN locations
        nan_mask = torch.isnan(analysis_tensor)
        if tensor.dim() <= 2:  # Only show for 1D and 2D tensors
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            if len(nan_indices) <= 10:  # Limit output
                logging.error(f"  NaN locations: {nan_indices.tolist()}")
    
    # Inf detection  
    inf_count = torch.isinf(analysis_tensor).sum().item()
    if inf_count > 0:
        logging.error(f"  Inf values: {inf_count}/{tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)")
        
        # Find Inf locations
        inf_mask = torch.isinf(analysis_tensor)
        if tensor.dim() <= 2:  # Only show for 1D and 2D tensors
            inf_indices = torch.nonzero(inf_mask, as_tuple=False)
            if len(inf_indices) <= 10:  # Limit output
                logging.error(f"  Inf locations: {inf_indices.tolist()}")
    
    # Statistical analysis for non-problematic tensors
    if nan_count == 0 and inf_count == 0:
        try:
            tensor_min = analysis_tensor.min().item()
            tensor_max = analysis_tensor.max().item()
            tensor_mean = analysis_tensor.mean().item()
            tensor_std = analysis_tensor.std().item()
            
            logging.info(f"  Min: {tensor_min:.6f}")
            logging.info(f"  Max: {tensor_max:.6f}")
            logging.info(f"  Mean: {tensor_mean:.6f}")
            logging.info(f"  Std: {tensor_std:.6f}")
            
            # Check for potential numerical instability
            if abs(tensor_max) > 1e6 or abs(tensor_min) > 1e6:
                logging.warning(f"  Large values detected - potential numerical instability")
            
            if tensor_std < 1e-8:
                logging.warning(f"  Very small standard deviation - potential gradient issues")
                
        except Exception as e:
            logging.error(f"  Could not compute statistics: {e}")


def fix_division_by_zero(numerator, denominator, epsilon=1e-8):
    """
    Perform safe division by adding a small epsilon to the denominator to avoid division by zero.
    
    Args:
        numerator (torch.Tensor): The numerator tensor
        denominator (torch.Tensor): The denominator tensor
        epsilon (float): Small value to add to denominator to prevent division by zero
        
    Returns:
        torch.Tensor: Result of safe division
    """
    if not isinstance(numerator, torch.Tensor):
        numerator = torch.tensor(numerator, dtype=torch.float32)
    
    if not isinstance(denominator, torch.Tensor):
        denominator = torch.tensor(denominator, dtype=torch.float32)
    
    # Ensure denominator is not exactly zero
    safe_denominator = torch.where(
        torch.abs(denominator) < epsilon,
        torch.sign(denominator) * epsilon,
        denominator
    )
    
    result = numerator / safe_denominator
    
    # Check if the operation introduced any numerical issues
    if torch.isnan(result).any() or torch.isinf(result).any():
        logging.warning("Safe division still resulted in NaN/Inf values")
        diagnose_tensor_issues(numerator, "numerator")
        diagnose_tensor_issues(denominator, "denominator") 
        diagnose_tensor_issues(result, "division_result")
    
    return result


def check_gradient_health(model, threshold=1e-6):
    """
    Check the health of gradients in a model.
    
    Args:
        model (torch.nn.Module): The model to check
        threshold (float): Threshold below which gradients are considered too small
        
    Returns:
        dict: Dictionary with gradient health statistics
    """
    stats = {
        'total_params': 0,
        'params_with_grad': 0,
        'nan_grads': 0,
        'inf_grads': 0,
        'small_grads': 0,
        'large_grads': 0,
        'grad_norm': 0.0
    }
    
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            stats['total_params'] += 1
            stats['params_with_grad'] += 1
            
            grad = param.grad.data
            
            # Check for NaN/Inf
            if torch.isnan(grad).any():
                stats['nan_grads'] += 1
                logging.warning(f"NaN gradient in parameter: {name}")
            
            if torch.isinf(grad).any():
                stats['inf_grads'] += 1
                logging.warning(f"Inf gradient in parameter: {name}")
            
            # Check gradient magnitude
            grad_norm = grad.norm().item()
            total_norm += grad_norm ** 2
            
            if grad_norm < threshold:
                stats['small_grads'] += 1
            elif grad_norm > 10.0:  # Arbitrary large threshold
                stats['large_grads'] += 1
                logging.warning(f"Large gradient in parameter {name}: {grad_norm}")
        else:
            stats['total_params'] += 1
    
    stats['grad_norm'] = total_norm ** 0.5
    
    return stats


def apply_gradient_clipping(model, max_norm=1.0):
    """
    Apply gradient clipping to prevent exploding gradients.
    
    Args:
        model (torch.nn.Module): The model whose gradients to clip
        max_norm (float): Maximum norm for gradient clipping
        
    Returns:
        float: The total norm of the gradients before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    if total_norm > max_norm:
        logging.warning(f"Gradients clipped: norm was {total_norm:.4f}, clipped to {max_norm}")
    
    return total_norm
