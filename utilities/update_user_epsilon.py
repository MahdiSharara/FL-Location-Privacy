import numpy as np
import logging

def update_user_epsilon(users, solution=None, cancel_privacy=False):
    """
    Update the epsilon values for each user. Simplified version for RL-only use.

    Args:
        users (dict[int, User]): Dictionary of User objects.
        solution (dict[str, np.ndarray]): Not used in the RL version.
        cancel_privacy (bool): If True, set epsilon to infinity to cancel privacy.
    """
    
    if cancel_privacy:
        # Set epsilon to infinity to cancel privacy
        for user_id, user in users.items():
            user.set_epsilon_generate_fake_location(float('inf'))
            logging.info(f"User ID: {user_id}, Cancel privacy, Updated Epsilon: inf, Max Epsilon: {user.max_epsilon}")
    else:
        # Set epsilon to the user's max epsilon (maximum privacy)
        for user_id, user in users.items():
            user.set_epsilon_generate_fake_location(user.max_epsilon)
            logging.info(f"User ID: {user_id}, setting to max Epsilon: {user.max_epsilon}")
