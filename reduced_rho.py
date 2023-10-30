"""
Author: Gonzalo Ferro
"""
import logging
import numpy as np
from itertools import product
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

def reduced_rho_mps_first(mps, free_indices, contraction_indices):
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
    """
    Secont try of computing reduced density matrix using MPS. In this
    case the function uses MAtrix Product Operators (MPO).  It uses
    mpo_contraction function. All the tensors involved will be
    rank-4 or rank-2. Have memory problems when lot of free indices
    are needed. I think the problem is the open boundaries for first
    and last MPO.
    """
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

def contraction_pl(tensor):
    tensor = contract_indices(tensor, tensor.conj(), [1], [1])
    tensor = tensor.transpose(0, 2, 1, 3)
    reshape = [
        tensor.shape[0] * tensor.shape[1],
        tensor.shape[2] * tensor.shape[3]
    ]
    tensor = tensor.reshape(reshape)
    return tensor

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

def reduced_rho_mps_01(mps, free_indices, contraction_indices):
    """
    Computes reduced density matrix by contracting first the MPS
    with NON contracted legs. Then compute the contraction of the
    MPS with contracted legs. Finally Contract the result of the 
    Non Contracted operations with the Contracted Ones for getting
    the desired reduced density matrix. Try to alternate MPS contraction
    of states (free legs) with MPO contractions (contracted legs)
    """
    # First deal with contraction indices
    tensor_contracted = contraction_pl(mps[contraction_indices[0]])
    for i in contraction_indices[1:]:
        #print(i)
        tensor = contraction_pl(mps[i])
        tensor_contracted = mpo_contraction(tensor_contracted, tensor)
    # tensor_contracted is a matrix formed with the tensors with
    # contracted physical legs

    # Second deal with free indices
    tensor_free = mps[free_indices[0]]
    for i in free_indices[1:]:
        #print(i)
        tensor = mps[i]
        #print(tensor.shape)
        tensor_free= contract_indices(tensor_free, tensor, [2], [0])
        #print(tensor_free.shape)
        reshape = [
            tensor_free.shape[0] ,
            tensor_free.shape[1] * tensor_free.shape[2],
            tensor_free.shape[3],
        ]
        tensor_free = tensor_free.reshape(reshape)
    logger.debug("Free tensor: {}".format(tensor_free.shape))
    # tensor free is the result of the contractions of tensors with
    # non contracted physical legs. But IT is a state not a density matrix

    # From the free index tensors we created the corresponding rho
    tensor_free = contract_indices(tensor_free, tensor_free.conj(), [], [])
    tensor_free = tensor_free.transpose(0, 3, 1, 4, 2, 5)
    reshape = [
        tensor_free.shape[0] * tensor_free.shape[1],
        tensor_free.shape[2],  tensor_free.shape[3],
        tensor_free.shape[4] * tensor_free.shape[5]
    ]
    tensor_free = tensor_free.reshape(reshape)
    logger.debug("Contracted tensor: {}".format(tensor_contracted.shape))
    logger.debug("Free tensor: {}".format(tensor_free.shape))

    # Contracted the rho with the rest of the matrix
    tensor_out = contract_indices(tensor_free, tensor_contracted, [3, 0], [0, 1])
    return tensor_out

def reduced_rho_mps(mps, free_indices, contraction_indices):
    """
    Computes reduced density matrix by contracting first the MPS
    with NON contracted legs. Then compute the contraction of the
    MPS with contracted legs. Finally Contract the result of the
    Non Contracted operations with the Contracted Ones for getting
    the desired reduced density matrix. Try to alternate MPS contraction
    of states (free legs) with MPO contractions (contracted legs)
    """
    # First deal with contraction indices
    logger.debug("MPS: {}".format([a.shape for a in mps]))
    tensor_contracted = contraction_pl(mps[contraction_indices[0]])
    for i in contraction_indices[1:]:
        #print(i)
        tensor = contraction_pl(mps[i])
        tensor_contracted = mpo_contraction(tensor_contracted, tensor)
    # tensor_contracted is a matrix formed with the tensors with
    # contracted physical legs

    first_dim = int(np.sqrt(tensor_contracted.shape[0]))
    second_dim = int(np.sqrt(tensor_contracted.shape[-1]))
    tensor_contracted = tensor_contracted.reshape([
        first_dim, first_dim, second_dim,second_dim])

    # Second deal with free indices
    tensor_free = mps[free_indices[0]]
    for i in free_indices[1:]:
        #print(i)
        tensor = mps[i]
        #print(tensor.shape)
        tensor_free= contract_indices(tensor_free, tensor, [2], [0])
        #print(tensor_free.shape)
        reshape = [
            tensor_free.shape[0] ,
            tensor_free.shape[1] * tensor_free.shape[2],
            tensor_free.shape[3],
        ]
        tensor_free = tensor_free.reshape(reshape)
    logger.debug("Free tensor: {}".format(tensor_free.shape))
    logger.debug("Contracted tensor: {}".format(tensor_contracted.shape))
    # tensor free is the result of the contractions of tensors with
    # non contracted physical legs
    tensor_out = contract_indices(
        tensor_free, tensor_contracted, [2, 0], [0, 2])
    logger.debug("Output tensor: {}".format(tensor_out.shape))
    tensor_out = contract_indices(
        tensor_out, tensor_free.conj(), [1, 2], [2, 0])
    logger.debug("Output tensor: {}".format(tensor_out.shape))

    return tensor_out

def reduced_rho_mps_test(mps, free_indices, contraction_indices):
    """
    Computes reduced density matrix by contracting first the MPS
    with NON contracted legs. Then compute the contraction of the
    MPS with contracted legs. Finally Contract the result of the
    Non Contracted operations with the Contracted Ones for getting
    the desired reduced density matrix. Try to alternate MPS contraction
    of states (free legs) with MPO contractions (contracted legs)
    """
    # First deal with contraction indices
    logger.debug("MPS: {}".format([a.shape for a in mps]))
    tensor_contracted = contraction_pl(mps[contraction_indices[0]])
    for i in contraction_indices[1:]:
        #print(i)
        tensor = contraction_pl(mps[i])
        #tensor_contracted = mpo_contraction(tensor_contracted, tensor)
        tensor_contracted = contract_indices(tensor_contracted, tensor, [2], [0])
        reshape = [
            tensor_contracted.shape[0] ,
            tensor_contracted.shape[1] * tensor_contracted.shape[2],
            tensor_contracted.shape[3],
        ]
        tensor_contracted = tensor_contracted.reshape(reshape)
    # tensor_contracted is a matrix formed with the tensors with
    # contracted physical legs

    first_dim = int(np.sqrt(tensor_contracted.shape[0]))
    second_dim = int(np.sqrt(tensor_contracted.shape[-1]))
    tensor_contracted = tensor_contracted.reshape([
        first_dim, first_dim, second_dim,second_dim])

    # Second deal with free indices
    tensor_free = mps[free_indices[0]]
    for i in free_indices[1:]:
        #print(i)
        tensor = mps[i]
        #print(tensor.shape)
        tensor_free= contract_indices(tensor_free, tensor, [2], [0])
        #print(tensor_free.shape)
        reshape = [
            tensor_free.shape[0] ,
            tensor_free.shape[1] * tensor_free.shape[2],
            tensor_free.shape[3],
        ]
        tensor_free = tensor_free.reshape(reshape)
    logger.debug("Free tensor: {}".format(tensor_free.shape))
    logger.debug("Contracted tensor: {}".format(tensor_contracted.shape))
    # tensor free is the result of the contractions of tensors with
    # non contracted physical legs

    # First Deal with Contracted
    tensor_contracted = contract_indices(
        tensor_contracted, tensor_contracted.conj(), [1], [1])
    tensor_contracted = tensor_contracted.transpose(0, 2, 1, 3)
    logger.debug("Contracted tensor: {}".format(tensor_contracted.shape))

    tensor_out = contract_indices(
        tensor_free, tensor_contracted, [2, 0], [0, 2])
    logger.debug("Output tensor: {}".format(tensor_out.shape))
    tensor_out = contract_indices(
        tensor_out, tensor_free.conj(), [1, 2], [2, 0])
    logger.debug("Output tensor: {}".format(tensor_out.shape))

    return tensor_out
