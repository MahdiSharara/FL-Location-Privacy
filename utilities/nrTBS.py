from scipy.io import loadmat
import numpy as np
import os
from utilities.load_json import load_config
config = load_config('config.json')
num_RBs = config["n_RBs"]
num_MCS = config["n_MCS"]
# The indexes of the considered MCS (Modulation and Coding Scheme) values.
# Define the arrays
tcr1 = np.array([78, 120, 193, 308, 449, 602, 378, 490, 616, 466, 567, 666, 772, 873, 948]) / 1024
tcr2 = np.array([120, 157, 193, 251, 308, 379, 449, 526, 602, 679, 340, 378, 434, 490,
                 553, 616, 658, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948]) / 1024

# Find indices in tcr2 where elements of tcr1 exist
considered_MCS = [np.where(tcr2 == val)[0][0] for val in tcr1 if val in tcr2]
del tcr1, tcr2

#considered_MCS = [0, 2, 4, 6, 8, 10, 12, 14 ,16, 18, 22, 26]

def load_nrTBSMatrix(mat_name='nrTBSMatrix.mat', var_name='nrTBSMatrix',nRB=num_RBs):
    """
    Load the nrTBSMatrix from a .mat file.
    
    Returns:
        nrTBSMatrix (numpy.ndarray): nRB x MCS : The loaded nrTBSMatrix in Mbps
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mat_name = os.path.join(script_dir, mat_name)
    # Load the .mat file
    data = loadmat(mat_name)
    nrTBSMatrix = data[var_name]
    # Check if the variable exists in the loaded data   
    if var_name not in data:
        raise ValueError(f"Variable '{var_name}' not found in the .mat file.")
    # Check if the variable is a numpy array    
    if not isinstance(nrTBSMatrix, np.ndarray):
        raise ValueError(f"Variable '{var_name}' is not a numpy array.")
    # Check if the variable is a 2D array
    if len(nrTBSMatrix.shape) != 2:
        raise ValueError(f"Variable '{var_name}' is not a 2D array.")
    # Check if the variable is empty
    if nrTBSMatrix.size == 0:
        raise ValueError(f"Variable '{var_name}' is empty.")
    #output a dictoinary    

    # Save the dictionary as a .npy file
    #np.save('nrTBSMatrix_dict.npy', nrTBS_dict)
    #return matrix and dictionary using the considered MCS values only
    nrTBSMatrix = nrTBSMatrix[:, considered_MCS]
    nrTBS_dict = {(i+1, considered_MCS[j]): nrTBSMatrix[i, j] for i in range(nrTBSMatrix.shape[0]) for j in range(nrTBSMatrix.shape[1])}  # return dictionary using the considered MCS values not all the matrix. The MCS index can be 0, 2, 4, 6, 8, 10, 12, 14 ,16, 18, 22, 26
    nrTBSMatrix = nrTBSMatrix[:num_RBs, :num_MCS]  # Select the first nRB rows and nMCS columns
    
    return nrTBSMatrix, nrTBS_dict
if __name__ == "__main__":
    nrTBSMatrix, nrTBS_dict =load_nrTBSMatrix(mat_name='nrTBSMatrix.mat', var_name='nrTBSMatrix')
    
# mat_name='nrTBSMatrix.mat'
# var_name='nrTBSMatrix'

# script_dir = os.path.dirname(os.path.abspath(__file__))
# print(script_dir)
#     # Go to parent directory (project root)
# parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
# print(parent_dir)
#     # Full path to the .mat file
# mat_path = os.path.join(script_dir, mat_name)
# print(mat_path)

# nrTBSMatrix, nrTBS_dict = load_nrTBSMatrix(mat_name=mat_path, var_name='nrTBSMatrix')


