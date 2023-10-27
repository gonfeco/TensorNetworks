"""
Here there are functions that performs contractions on a MPS that does
not work for computing reduced density matrix in translational invariant
MPPS with periodic boundary conditions

Author: Gonzalo Ferro
"""
import logging
from itertools import product
import numpy as np
from tensornetworks import contract_indices, contract_indices_one_tensor
logger = logging.getLogger('__name__')

def mpo_contraction_0(tensor_1, tensor_2):
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

def reduced_rho_mpo_fulllist(mps, free_indices, contraction_indices):
    """
    Computes reduced rho density matrix by doing pure matrix
    matrix multiplication of different indices of the physical legs
    of the MPS. It works  but need lot of time for computing reduced
    density matrices with lot of free indices
    """
    opa = list(product([0, 1], repeat=2 * len(free_indices)))
    reduced_rho = np.zeros(
        (2**len(free_indices), 2**len(free_indices)), dtype="complex")
    logger.debug("opa: {}".format(len(opa)))
    for step in opa:
        tensor_out = mps[0]
        if 0 in free_indices:
            i = free_indices.index(0)
            tensor_out = np.kron(
                tensor_out[:, step[2*i], :],
                tensor_out.conj()[:, step[2*i+1], :])
        else:
            tensor_out = contraction_pl(tensor_out)
        for i in range(1, len(mps)):
            tensor = mps[i]
            if i in free_indices:
                j = free_indices.index(i)
                tensor = np.kron(
                    tensor[:, step[2*j], :], tensor.conj()[:, step[2*j+1], :])
            elif i in contraction_indices:
                tensor = contraction_pl(tensor)
            tensor_out = tensor_out @ tensor
        #logger.debug("Shape tensor_out out: %s", tensor_out.shape)
        row = [v for i, v in enumerate(step) if i%2==0]
        row_int = sum([v * 2 ** i for i, v in enumerate(row)])
        #print(row, row_int)
        column = [v for i, v in enumerate(step) if i%2!=0]
        column_int = sum([v * 2 ** i for i, v in enumerate(column)])
        #print(column, column_int)
        reduced_rho[row_int, column_int] = np.trace(tensor_out)
    return reduced_rho

def contraction_pl(tensor):
    tensor = contract_indices(tensor, tensor.conj(), [1], [1])
    tensor = tensor.transpose(0, 2, 1, 3)
    reshape = [
        tensor.shape[0] * tensor.shape[1],
        tensor.shape[2] * tensor.shape[3]
    ]
    tensor = tensor.reshape(reshape)
    return tensor
