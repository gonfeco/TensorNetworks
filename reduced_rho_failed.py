"""
Here there are functions that performs contractions on a MPS that does
not work for computing reduced density matrix in translational invariant
MPPS with periodic boundary conditions

Author: Gonzalo Ferro
"""
import logging
import numpy as np
from tensornetworks import contract_indices, contract_indices_one_tensor
logger = logging.getLogger('__name__')

def mpo_contraction_failed(tensor_1, tensor_2):
    """
    Contraction of 2 input tensors with corresponding adjust of dimension
    for computing density matrices.

    Parameters
    ----------

    tensor_1 : np array
        First input 6 or 4 rank tensor
    tensor_2 : np array
        Second input 6 or 4 rank tensor

    Returns
    _______

    step : np array
        output rank 6 or 4 tensor
    """

    rank_tensor_1 = tensor_1.ndim
    rank_tensor_2 = tensor_2.ndim

    if (rank_tensor_1 == 3) and (rank_tensor_2 == 4):
        # Case 0
        tensor_out = contract_indices(tensor_1, tensor_2, [2], [0])
        tensor_out = tensor_out.transpose(0, 2, 1, 3, 4)
        reshape = [
           tensor_out.shape[0] * tensor_out.shape[1],
           tensor_out.shape[2] * tensor_out.shape[3],
           tensor_out.shape[4]
        ]
        tensor_out = tensor_out.reshape(reshape)
    elif (rank_tensor_1 == 3) and (rank_tensor_2 == 2):
        # Case 1
        tensor_out = contract_indices(tensor_1, tensor_2, [2], [0])
    elif (rank_tensor_1 == 1) and (rank_tensor_2 == 4):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [0], [0])
    elif (rank_tensor_1 == 1) and (rank_tensor_2 == 2):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [0], [0])
    elif (rank_tensor_1 == 3) and (rank_tensor_2 == 3):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [2], [0])
        tensor_out = tensor_out.transpose(0, 2, 1, 3)
        reshape = [
           tensor_out.shape[0] * tensor_out.shape[1],
           tensor_out.shape[2] * tensor_out.shape[3]
        ]
        tensor_out = tensor_out.reshape(reshape)
    elif (rank_tensor_1 == 3) and (rank_tensor_2 == 1):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [2], [0])
    else:
        raise ValueError("Input Tensors MUST be rank-4 or rank-6")
    return tensor_out


