"""
Author: Gonzalo Ferro
"""
import logging
import numpy as np
from tensornetworks import contract_indices, contract_indices_one_tensor
logger = logging.getLogger('__name__')

def density_matrix_mps_contracion(tensor_1, tensor_2):
    """
    Contraction of 2 input tensors with corresponding adjust of dimension
    for computing density matrices.
    The input tensors MUST be rank 6 or 4 tensors.

    Rank-6 tensor:    |  Rank-4 tensor:
        1             |
    0 - |   - 2       |  0 - - 1
         o            |     o
    3 -   | - 5       |  2 - - 3
          4           |

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

    if (rank_tensor_1 == 6) and (rank_tensor_2 == 4):
        # Case 0
        tensor_out = contract_indices(tensor_1, tensor_2, [2, 5], [0, 2])
        tensor_out = tensor_out.transpose(0, 1, 4, 2, 3, 5)
    elif (rank_tensor_1 == 4) and (rank_tensor_2 == 4):
        # Case 1
        tensor_out = contract_indices(tensor_1, tensor_2, [1, 3], [0, 2])
        tensor_out = tensor_out.transpose(0, 2, 1, 3)
    elif (rank_tensor_1 == 4) and (rank_tensor_2 == 6):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [1, 3], [0, 3])
        tensor_out = tensor_out.transpose(0, 2, 3, 1, 4, 5)
    elif (rank_tensor_1 == 6) and (rank_tensor_2 == 6):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [2, 5], [0, 3])
        tensor_out = tensor_out.transpose(0, 2, 1, 4, 3, 6, 5, 7)
        reshape = [
            tensor_out.shape[0], tensor_out.shape[1],
            tensor_out.shape[2] * tensor_out.shape[3],
            tensor_out.shape[4] * tensor_out.shape[5],
            tensor_out.shape[6], tensor_out.shape[7]
        ]
        tensor_out = tensor_out.reshape(reshape)
        tensor_out = tensor_out.transpose(0, 2, 4, 1, 3, 5)
    else:
        raise ValueError("Input Tensors MUST be rank-4 or rank-6")
    return tensor_out

def reduced_rho_mps_firs(mps, free_indices, contraction_indices):
    """
    First try of computing reduced density matrix using MPS. It uses
    density_matrix_mps_contracion. All the tensors involved will be
    rank-6 or rank-4. Have memory problems when lot of free indices
    are needed.
    """
    i = 0
    tensor_out = mps[i]
    # Starting Tensor for Denisty Matrix
    if i in free_indices:
        tensor_out = contract_indices(tensor_out, tensor_out.conj(), [], [])
    elif i in contraction_indices:
        tensor_out = contract_indices(tensor_out, tensor_out.conj(), [1], [1])
    else:
        raise ValueError("Problem with site i: {}".format(i))
    for i in range(1, len(mps)):
        tensor = mps[i]
        if i in free_indices:
            tensor = contract_indices(tensor, tensor.conj(), [], [])
        elif i in contraction_indices:
            tensor = contract_indices(tensor, tensor.conj(), [1], [1])
        else:
            raise ValueError("Problem with site i: {}".format(i))
        tensor_out = density_matrix_mps_contracion(tensor_out, tensor)
        logger.debug("\t tensor_out.shape: %s", tensor_out.shape)
    tensor_out = contract_indices_one_tensor(tensor_out, [(0, 2), (3, 5)])
    return tensor_out

def mpo_contraction(tensor_1, tensor_2):
    """
    Contraction of 2 input tensors (npo tensors) with corresponding
    adjust of dimension for computing density matrices.
    The input tensors MUST be rank 4 or 2 tensors.

    Rank-4 tensor:    |  Rank-2 tensor:
         |            |
       - o -          |     - o -
         |            |

    Parameters
    ----------

    tensor_1 : np array
        First input 4 or 2 rank tensor
    tensor_2 : np array
        Second input 4 or 2 rank tensor

    Returns
    _______


    step : np array
        output rank 4 or 2 tensor
    """

    rank_tensor_1 = tensor_1.ndim
    rank_tensor_2 = tensor_2.ndim

    if (rank_tensor_1 == 4) and (rank_tensor_2 == 4):
        # Case 0
        tensor_out = contract_indices(tensor_1, tensor_2, [3],[0])
        tensor_out = tensor_out.transpose(0, 1, 3, 2, 4, 5)
        reshape = [
            tensor_out.shape[0],
            tensor_out.shape[1] * tensor_out.shape[2],
            tensor_out.shape[3] * tensor_out.shape[4],
            tensor_out.shape[5]
        ]
        tensor_out = tensor_out.reshape(reshape)
    elif (rank_tensor_1 == 4) and (rank_tensor_2 == 2):
        # Case 1
        tensor_out = contract_indices(tensor_1, tensor_2, [3], [0])
    elif (rank_tensor_1 == 2) and (rank_tensor_2 == 4):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [1], [0])
    elif (rank_tensor_1 == 2) and (rank_tensor_2 == 2):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [1], [0])
    else:
        raise ValueError("Input Tensors MUST be rank-4 or rank-2")
    return tensor_out

def reduced_rho_mpo_first(mps, free_indices, contraction_indices):
    logger.debug("mps: {}".format([a.shape for a in mps]))
    i = 0
    tensor_out = mps[i]
    # Starting Tensor for Denisty Matrix
    tensor = mps[0]
    if i in free_indices:
        tensor_out = contract_indices(tensor_out, tensor_out.conj(), [], [])
        tensor_out = tensor_out.transpose(0, 3, 1, 4, 2, 5)
        reshape = [
            tensor_out.shape[0] * tensor_out.shape[1],
            tensor_out.shape[2], tensor_out.shape[3],
            tensor_out.shape[4] * tensor_out.shape[5],
        ]
        tensor_out = tensor_out.reshape(reshape)
    elif i in contraction_indices:
        tensor_out = contract_indices(tensor_out, tensor_out.conj(), [1], [1])
        tensor_out = tensor_out.transpose(0, 2, 1, 3)
        reshape = [
            tensor_out.shape[0] * tensor_out.shape[1],
            tensor_out.shape[2] * tensor_out.shape[3]
        ]
        tensor_out = tensor_out.reshape(reshape)
    else:
        raise ValueError("Problem with site i: {}".format(i))
    for i in range(1, len(mps)):
        tensor = mps[i]
        if i in free_indices:
            tensor = contract_indices(tensor, tensor.conj(), [], [])
            tensor = tensor.transpose(0, 3, 1, 4, 2, 5)
            reshape = [
                tensor.shape[0] * tensor.shape[1],
                tensor.shape[2], tensor.shape[3],
                tensor.shape[4] * tensor.shape[5],
            ]
            tensor = tensor.reshape(reshape)

        elif i in contraction_indices:
            tensor = contract_indices(tensor, tensor.conj(), [1], [1])
            tensor = tensor.transpose(0, 2, 1, 3)
            reshape = [
                tensor.shape[0] * tensor.shape[1],
                tensor.shape[2] * tensor.shape[3]
            ]
            tensor = tensor.reshape(reshape)
        else:
            raise ValueError("Problem with site i: {}".format(i))
        logger.debug("Shape tensor_out in: %s", tensor_out.shape)
        #logger.debug("Shape tensor: %s", tensor.shape)
        tensor_out = mpo_contraction(tensor_out, tensor)
        logger.debug("Shape tensor_out out: %s", tensor_out.shape)
    tensor_out = contract_indices_one_tensor(tensor_out, [(0, 3)])
    return tensor_out



