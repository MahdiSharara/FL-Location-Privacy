import numpy as np
used_mcs_indexes = [0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27]

def update_user_parameters(users, solution, nrTBSMatrix):
    """
    Update the parameters of each user based on the RL allocation results.
    
    Args:
        users (dict[int, User]): Dictionary of User objects.
        solution (dict[str, np.ndarray]): Dictionary of solution arrays from the RL allocation.
        nrTBSMatrix (np.ndarray or dict): The TBS matrix for calculating throughput.
    """
    try:
        # RL solution uses 'z' for BS assignment
        associated_BS = np.where(solution['z'].sum(axis=1) == 0, -1, np.argmax(solution['z'], axis=1))

        # Calculate associated server for each user
        associated_server = np.where(solution['y'].sum(axis=1) == 0, -1, np.argmax(solution['y'], axis=1))
        is_served = solution['y'].sum(axis=1) == 1
        
        # Calculate resource allocations by finding RBs and MCS for each user
        selected_nRBs = np.zeros(len(users), dtype=int) - 1  # Default to -1 (not served)
        selected_mcs = np.zeros(len(users), dtype=int) - 1   # Default to -1 (not served)
        
        for user_id in range(len(users)):
            if is_served[user_id]:
                # Count non-zero elements for this user across all RBs and MCS
                rb_count_per_mcs = solution['zeta'][user_id].sum(axis=0)  # Sum over RBs for each MCS
                if rb_count_per_mcs.sum() > 0:  # If any allocation exists
                    # Find the MCS being used
                    mcs_index = np.argmax(rb_count_per_mcs)
                    # Count RBs for that MCS
                    rb_count = int(rb_count_per_mcs[mcs_index])
                    
                    selected_nRBs[user_id] = rb_count - 1  # Store as index (0-based)
                    selected_mcs[user_id] = mcs_index

        # Calculate throughput for each user
        throughput = np.zeros(len(users))
        for user_id in range(len(users)):
            rb_count = selected_nRBs[user_id]
            mcs = selected_mcs[user_id]
            
            if rb_count >= 0 and mcs >= 0:
                # Handle the nrTBSMatrix whether it's a numpy array or a dictionary
                if isinstance(nrTBSMatrix, dict):
                    # If it's a dictionary with tuple keys (rb_count, mcs)
                    throughput[user_id] = nrTBSMatrix.get((rb_count + 1, mcs), 0)
                else:
                    # If it's a numpy array with [rb_count, mcs] indexing
                    if rb_count < nrTBSMatrix.shape[0] and mcs < nrTBSMatrix.shape[1]:
                        throughput[user_id] = nrTBSMatrix[rb_count, mcs]

        # Calculate delay for each user
        achieved_delay = solution['T_trans'] + solution['T_proc']  # Sum transmission and processing delays

        # Update user properties
        for user_id, user in users.items():
            user.set_user_properties(
                associated_BS=associated_BS[user_id],
                associated_server=associated_server[user_id],
                assigned_RBs=(selected_nRBs[user_id] + 1) if selected_nRBs[user_id] != -1 else 0,
                assigned_MCS=used_mcs_indexes[selected_mcs[user_id]] if selected_mcs[user_id] != -1 else -1,
                achieved_throughput=throughput[user_id],
                achieved_delay=achieved_delay[user_id],
                is_served=(is_served[user_id] > 0)
            )
    except Exception as e:
        print(f"Error updating user parameters: {e}")