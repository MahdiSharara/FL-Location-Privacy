import json5
import os

def load_config(file_path: str) -> dict:
    """
    Load and validate the configuration from a JSON5 file.

    Args:
        file_path (str): Path to the JSON5 configuration file.

    Returns:
        dict: Loaded and validated configuration dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, 'r') as file:
            config = json5.load(file)
    except json5.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON5 configuration file: {e}")

    # Validate configuration
    required_keys = [
        "x_y_location_range", "z_location_range", "processing_capacity_range",
        "link_capacity_range", "n_RBs", "delay_requirement", "rate_requirement", "n_MCS", "noise_figure",
        "noise_spectral_density", "nrTBSMatrix_file", "nrTBSMatrix_variable",
        "P_max_Tx_dBm", "B_RB_Hz", "M_big", "gamma_th", "objective_weights", "deltaF"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Additional validation
    if not isinstance(config["x_y_location_range"], list) or len(config["x_y_location_range"]) != 2:
        raise ValueError("Invalid value for 'x_y_location_range'. Expected a list of two floats.")
    if not isinstance(config["z_location_range"], list) or len(config["z_location_range"]) != 2:
        raise ValueError("Invalid value for 'z_location_range'. Expected a list of two floats.")
    if not isinstance(config["processing_capacity_range"], list) or len(config["processing_capacity_range"]) != 2:
        raise ValueError("Invalid value for 'processing_capacity_range'. Expected a list of two floats.")
    
    if not isinstance(config["link_capacity_range"], list) or len(config["link_capacity_range"]) != 2:
        raise ValueError("Invalid value for 'link_capacity_range'. Expected a list of two floats.")
    if not isinstance(config["n_RBs"], int):
        raise ValueError("Invalid value for 'n_RBs'. Expected an integer.")
    if not isinstance(config["delay_requirement"], list) or len(config["delay_requirement"]) != 2:
        raise ValueError("Invalid value for 'delay_requirement'. Expected a list of two floats.")
    if not isinstance(config["rate_requirement"], list) or len(config["rate_requirement"]) != 2:
        raise ValueError("Invalid value for 'rate_requirement'. Expected a list of two floats.")
    if not isinstance(config["n_MCS"], int):
        raise ValueError("Invalid value for 'n_MCS'. Expected an integer.")
    if not isinstance(config["noise_figure"], (int, float)):
        raise ValueError("Invalid value for 'noise_figure'. Expected a float.")
    if not isinstance(config["noise_spectral_density"], (int, float)):
        raise ValueError("Invalid value for 'noise_spectral_density'. Expected a float.")
    if not isinstance(config["nrTBSMatrix_file"], str):
        raise ValueError("Invalid value for 'nrTBSMatrix_file'. Expected a string.")
    if not isinstance(config["nrTBSMatrix_variable"], str):
        raise ValueError("Invalid value for 'nrTBSMatrix_variable'. Expected a string.")
    if not isinstance(config["P_max_Tx_dBm"], (int, float)):
        raise ValueError("Invalid value for 'P_max_Tx_dBm'. Expected a float.")
    if not isinstance(config["B_RB_Hz"], (int, float)):
        raise ValueError("Invalid value for 'B_RB_Hz'. Expected a float.")
    if not isinstance(config["M_big"], (int, float)):
        raise ValueError("Invalid value for 'M_big'. Expected a float.")
    if not isinstance(config["gamma_th"], list) or len(config["gamma_th"]) != 14:
        raise ValueError("Invalid value for 'gamma_th'. Expected a list of 14 floats.")
    if not isinstance(config["objective_weights"], dict):
        raise ValueError("Invalid value for 'objective_weights'. Expected a dictionary.")
    if not isinstance(config["deltaF"], (int, float)):
        raise ValueError("Invalid value for 'deltaF'. Expected a float.")

    return config