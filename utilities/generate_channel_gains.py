import numpy as np
import matplotlib.pyplot as plt

def generate_channel_gains(d_real: np.ndarray, f: float, n_RBs: int, alpha: float = 3.6, beta: float = 7.6, gamma: float = 2, sigma_sf: float = 9.4, bandwidth: float = 180000, noise_figure_db: float = 8) -> np.ndarray:
    """
    Generate channel gains based on the ABG path loss model and calculate the noise power considering the noise figure.

    Args:
        d_real (np.ndarray): Real distances between users and nodes (in meters).
        f (float): Carrier frequency (in GHz).
        alpha (float): Path loss exponent (default: 3.5).
        beta (float): Path loss offset in dB (default: 25).
        gamma (float): Frequency-dependent path loss exponent (default: 2).
        sigma_sf (float): Standard deviation of shadowing in dB (default: 7.6).
        bandwidth (float): Bandwidth in Hz (default: 1 MHz).
        noise_figure_db (float): Noise figure in dB (default: 8).

    Returns:
        np.ndarray: Channel gains for each user-node pair and noise power in linear scale.
    """    # Constants
    noise_spectral_density_dbm_hz = -174  # Noise spectral density in dBm/Hz

    # Ensure distances are positive to prevent log(0) or log(negative)
    d_safe = np.maximum(d_real, 1e-3)  # Minimum distance of 1mm
    
    # Path loss calculation based on ABG model
    path_loss = 10 * alpha * np.log10(d_safe / 1) + beta + 10 * gamma * np.log10(f / 1)  # d in meters, f in GHz

    # Generate unique shadowing values for each RB
    shadowing = np.random.normal(0, sigma_sf, size=(d_real.shape[0], d_real.shape[1], n_RBs))

    # Generate unique fading values for each RB
    fading = np.random.exponential(1, size=(d_real.shape[0], d_real.shape[1], n_RBs))
    # Convert path loss and shadowing to linear scale and calculate channel gains
    channel_gains = np.power(10, -(path_loss[..., None] + shadowing) / 10) * fading    # Calculate noise power in linear scale, including the noise figure
    # Ensure bandwidth is positive
    bandwidth_safe = max(bandwidth, 1.0)  # Minimum bandwidth of 1 Hz
    noise_power_dbm = noise_spectral_density_dbm_hz + 10 * np.log10(bandwidth_safe) + noise_figure_db
    noise_power_linear = np.power(10, noise_power_dbm / 10)  # Convert dBm to linear scale

    return channel_gains, noise_power_linear

if __name__ == "__main__":
    # Parameters
    nOfUsers = 1
    nOfBSs = 1
    nOfRBs = 1
    nRuns = 500000

    # Fixed positions
    usersPositions = np.array([[0, 0, 0]])  # User at origin
    BSPositions = np.array([[2000, 0, 0]])  # BS 2000 meters away

    # Storage for channel gain
    channelGains = np.zeros(nRuns)
    distances = np.zeros(nRuns)

    for i in range(nRuns):
        # Generate channel conditions
        distanceUsersToBSs = np.sqrt(np.sum((usersPositions - BSPositions) ** 2, axis=1))
        d_real = np.expand_dims(distanceUsersToBSs, axis=(0, 1))  # Shape (1, 1)

        channelGain, _ = generate_channel_gains(d_real, f=2, n_RBs=nOfRBs)
        channelGains[i] = np.mean(channelGain)
        distances[i] = np.mean(distanceUsersToBSs)    # Compute statistics
    avgGainLinear = np.mean(channelGains)
    # Protect against log(0) or log(negative)
    avgGain_dB = 10 * np.log10(max(avgGainLinear, 1e-12))

    # Display results
    print(f"Average Distance: {np.mean(distances):.2f} meters")
    print(f"Average Channel Gain: {avgGainLinear:.2e} (linear)")
    print(f"Average Channel Gain: {avgGain_dB:.2f} dB")
