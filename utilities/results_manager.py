"""
Centralized results management and analysis utilities.
Handles result saving, loading, and common analysis patterns.
"""

import os
import json
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from utilities.analyze_results_tools import analyze_results_multiple

logger = logging.getLogger(__name__)


class ResultsManager:
    """Centralized results management for simulation and RL experiments."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the results manager.
        
        Args:
            base_dir: Base directory for storing results. If None, uses current directory.
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def create_experiment_directory(self, experiment_name: str, timestamp: str = None) -> str:
        """
        Create a directory for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            timestamp: Optional timestamp. If None, current time is used.
            
        Returns:
            Path to the created directory
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        experiment_dir = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def save_experiment_config(self, config: dict, experiment_dir: str):
        """Save experiment configuration."""
        config_path = os.path.join(experiment_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_results(self, results: Dict, experiment_dir: str, filename: str = "results.json"):
        """
        Save results to a JSON file.
        
        Args:
            results: Results dictionary
            experiment_dir: Directory to save results
            filename: Name of the results file
        """
        results_path = os.path.join(experiment_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def load_results(self, results_path: str) -> Dict:
        """Load results from a JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def save_dataframe(self, df: pd.DataFrame, experiment_dir: str, filename: str):
        """Save a pandas DataFrame as CSV."""
        csv_path = os.path.join(experiment_dir, filename)
        df.to_csv(csv_path, index=False)
        logger.info(f"DataFrame saved to {csv_path}")
    
    def aggregate_multiple_runs(self, run_results: List[Dict], confidence_level: float = 0.95) -> Dict:
        """
        Aggregate results from multiple runs using statistical analysis.
        
        Args:
            run_results: List of result dictionaries from multiple runs
            confidence_level: Confidence level for confidence intervals
            
        Returns:
            Aggregated results with mean and confidence intervals
        """
        return analyze_results_multiple(run_results, confidence_level)
    
    def save_aggregated_results(self, run_results: List[Dict], experiment_dir: str, 
                               confidence_level: float = 0.95):
        """Save aggregated results from multiple runs."""
        aggregated = self.aggregate_multiple_runs(run_results, confidence_level)
        self.save_results(aggregated, experiment_dir, "aggregated_results.json")
        
        # Also save as CSV for easy analysis
        df_results = self._convert_aggregated_to_dataframe(aggregated)
        self.save_dataframe(df_results, experiment_dir, "aggregated_results.csv")
    
    def clean_old_results(self, days_old: int = 30) -> int:
        """
        Clean up old result directories.
        
        Args:
            days_old: Number of days old for directories to be considered for cleanup
            
        Returns:
            Number of directories cleaned up
        """
        if not os.path.exists(self.results_dir):
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                # Check if directory starts with 'results_' (old format)
                if item.startswith('results_'):
                    dir_time = os.path.getctime(item_path)
                    if dir_time < cutoff_time:
                        logger.info(f"Cleaning up old results directory: {item}")
                        import shutil
                        shutil.rmtree(item_path)
                        cleaned_count += 1
        
        return cleaned_count
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get a summary of all experiments in the results directory."""
        experiments = []
        
        if not os.path.exists(self.results_dir):
            return pd.DataFrame()
        
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                experiment_info = {
                    'name': item,
                    'path': item_path,
                    'created': datetime.fromtimestamp(os.path.getctime(item_path)),
                    'modified': datetime.fromtimestamp(os.path.getmtime(item_path)),
                    'size_mb': self._get_directory_size(item_path) / (1024 * 1024)
                }
                
                # Check for specific result files
                config_path = os.path.join(item_path, "experiment_config.json")
                results_path = os.path.join(item_path, "results.json")
                aggregated_path = os.path.join(item_path, "aggregated_results.json")
                
                experiment_info['has_config'] = os.path.exists(config_path)
                experiment_info['has_results'] = os.path.exists(results_path)
                experiment_info['has_aggregated'] = os.path.exists(aggregated_path)
                
                experiments.append(experiment_info)
        
        return pd.DataFrame(experiments)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert object to be JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj
    
    def _convert_aggregated_to_dataframe(self, aggregated: Dict) -> pd.DataFrame:
        """Convert aggregated results to a DataFrame."""
        rows = []
        for metric, stats in aggregated.items():
            if isinstance(stats, dict) and 'mean' in stats:
                row = {
                    'metric': metric,
                    'mean': stats['mean'],
                    'ci_lower': stats['confidence_interval'][0] if 'confidence_interval' in stats else None,
                    'ci_upper': stats['confidence_interval'][1] if 'confidence_interval' in stats else None
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _get_directory_size(self, path: str) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, IOError):
            pass
        return total_size


# Global results manager instance
results_manager = ResultsManager()


def save_experiment_results(results: Dict, experiment_name: str, config: Dict = None) -> str:
    """
    Convenience function to save experiment results.
    
    Args:
        results: Results dictionary
        experiment_name: Name of the experiment
        config: Optional configuration to save
        
    Returns:
        Path to the experiment directory
    """
    experiment_dir = results_manager.create_experiment_directory(experiment_name)
    
    if config is not None:
        results_manager.save_experiment_config(config, experiment_dir)
    
    results_manager.save_results(results, experiment_dir)
    
    return experiment_dir


def save_multiple_run_results(run_results: List[Dict], experiment_name: str, 
                             config: Dict = None, confidence_level: float = 0.95) -> str:
    """
    Convenience function to save and aggregate multiple run results.
    
    Args:
        run_results: List of result dictionaries from multiple runs
        experiment_name: Name of the experiment
        config: Optional configuration to save
        confidence_level: Confidence level for confidence intervals
        
    Returns:
        Path to the experiment directory
    """
    experiment_dir = results_manager.create_experiment_directory(experiment_name)
    
    if config is not None:
        results_manager.save_experiment_config(config, experiment_dir)
    
    # Save individual run results
    for i, result in enumerate(run_results):
        results_manager.save_results(result, experiment_dir, f"run_{i}_results.json")
    
    # Save aggregated results
    results_manager.save_aggregated_results(run_results, experiment_dir, confidence_level)
    
    return experiment_dir
