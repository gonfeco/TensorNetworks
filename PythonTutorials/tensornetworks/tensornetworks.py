"""
Implementation of reduced density matrix computations
"""

import string
import logging
import numpy as np
import copy
logger = logging.getLogger('__name__')


def contract_indices(tensor1, tensor2, contraction1=[], contraction2=[]):
    """
    Compute the contraction of 2 input tensors for the input contraction
    indices.The computation is done by, transposing indices, reshapin
    and doing matrix multiplication. Tensor legs can be of different
    dimension.
    BE AWARE: Order in contraction indices it is very important:
    contraction1 = [1, 2, 5] contraction2 = [2, 0, 6] -> Contractions
    will be: [1-2, 2-0, 5-6]
    But if contraction2 = [0, 2, 6] then contraction
    will be: [1-0, 2-2,5-6]
    If both contraction indices are empty then the tensors will be glued

    Parameters
    ----------

    tensor1 : numpy array
        first tensor
    tensor2 : numpy array
        second tensor
    contraction1 : list
        contraction indices for first tensor
    contraction2 : list
        contraction indices for second tensor

    Returns
    _______

    rho : numpy array
        Desired reduced density operator in matrix form

    """

    if len(contraction1) != len(contraction2):
        raise ValueError("Different number of contraction indices!")
    indices1 = list(range(tensor1.ndim))
    indices2 = list(range(tensor2.ndim))

    # Free indices for tensors
    free_indices1 = [i for i in indices1 if i not in contraction1]
    free_indices2 = [i for i in indices2 if i not in contraction2]

    # Transpose elements
    tensor1_t = tensor1.transpose(free_indices1 + contraction1)
    tensor2_t = tensor2.transpose(free_indices2 + contraction2)

    # If free_indices are empty
    if len(free_indices1) == 0:
        free_indices_1 = 1
    else:
        free_indices_1 = np.prod(
            [tensor1.shape[i] for i in free_indices1])

    # If contraction1 is empty
    if len(contraction1) == 0:
        contraction_indices_1 = 1
    else:
        contraction_indices_1 = np.prod(
            [tensor1.shape[i] for i in contraction1])

    # tensor1_t_matrix = tensor1_t.reshape(
    #     free_indices_1,
    #     np.prod([tensor1.shape[i] for i in contraction1]),
    # )

    tensor1_t_matrix = tensor1_t.reshape(
        free_indices_1, contraction_indices_1)

    # If free_indices are empty
    if len(free_indices2) == 0:
        free_indices_2 = 1
    else:
        free_indices_2 = np.prod(
            [tensor2.shape[i] for i in free_indices2])

    # If contraction1 is empty
    if len(contraction2) == 0:
        contraction_indices_2 = 1
    else:
        contraction_indices_2 = np.prod(
            [tensor2.shape[i] for i in contraction2])

    # tensor2_t_matrix = tensor2_t.reshape(
    #     free_indices_2,
    #     np.prod([tensor2.shape[i] for i in contraction2]),
    # )

    tensor2_t_matrix = tensor2_t.reshape(
        free_indices_2, contraction_indices_2)

    # Do the bare matrix multiplication
    contraction_tensor = tensor1_t_matrix @ tensor2_t_matrix.T

    free_contraction = [tensor1.shape[i] for i in free_indices1] + \
        [tensor2.shape[i] for i in free_indices2]
    contraction_tensor = contraction_tensor.reshape(free_contraction)

    return contraction_tensor


def reduced_matrix(state, free_indices, contraction_indices):
    """
    Compute the reduced density matrix for the input contraction indices
    The computation is done by, transposing indices, reshapin and doing
    matrix multiplication.
    The legs of the state MUST BE of same dimension.

    Parameters
    ----------

    state : numpy array
        array in MPS format of the state for computing reduced density
        matrix
    free_indices : list
        Free indices of the MPS state (this is the qubit that will NOT
        be traced out)
    contraction_indices : list
        Indices of the MPS state that will be contracted to compute
        the correspondent reduced density matrix

    Returns
    _______

    rho : numpy array
        Desired reduced density operator in matrix form

    """
    if len(contraction_indices) + len(free_indices) != state.ndim:
        raise ValueError(
            "Dimensions of free_indices and contraction_indices not compatible\
            with dimension of state")
    # First Transpose indices
    transpose_state = state.transpose(free_indices+contraction_indices)
    # Second rearrange as a marix
    matrix_state = transpose_state.reshape(
        len(state) ** len(free_indices), len(state) ** len(contraction_indices)
    )
    logger.debug('matrix_state_shape: {}'.format(matrix_state.shape))
    # Third Matrix Multiplication
    rho = matrix_state @ np.conj(matrix_state).T
    return rho

def mps_qr_lr_decompose(tensor):
    """
    Decompose the input tensor as  mps using QR decomposition.

    Parameters
    ----------

    tensor : numpy array
        input tensor to be decomposed as MPS using QR decomposition

    Returns
    _______

    mps : list
        list where each element is the rank-3 tensor as numpy array
        of the MPS QR decomposition of the input tensor
    """
    
    # Compute rank of the tensor
    rank = tensor.ndim
    logger.debug('rank: {}'.format(rank))
    # Dimensions of the legs
    leg_dim = tensor.shape
    logger.debug('leg_dim: {}'.format(leg_dim))
    m_step = copy.deepcopy(tensor)
    mps = []
    d_alphas = [1]
    # Reshape to matrix
    m_step = m_step.reshape(leg_dim[0], np.prod(leg_dim[1:]))
    size = 0.0
    for leg in range(rank-1):
        # loop over legs
        logger.debug('leg: {} m_step: {}'.format(leg, m_step.shape))
        # QR decomposition
        q_step, r_step = np.linalg.qr(m_step, 'reduced')
        d_alphas.append(r_step.shape[0])
        logger.debug('leg: {} q_step: {}'.format(leg, q_step.shape))
        logger.debug('leg: {} r_step: {}'.format(leg, r_step.shape))
        # Convert q isometric matrix to rank-3 tensor
        # -q-r- -> -q-r-
        #           |
        q_step = q_step.reshape(-1, leg_dim[leg], r_step.shape[0])
        logger.debug('leg: {} q_step tensor: {}'.format(leg, q_step.shape))
        mps.append(q_step)
        # For the following step of the loop we provided the R part
        m_step = r_step.reshape(r_step.shape[0] * leg_dim[leg+1], -1)
        size = size + q_step.size
    # Last leg procces
    r_step = r_step.reshape(-1, leg_dim[-1], 1)
    size = size + r_step.size
    logger.debug('leg: {} q_step tensor: {}'.format(rank-1, r_step.shape))
    mps.append(r_step)
    return mps, d_alphas, size


def mps_svd_lr_decompose(tensor):
    """
    Decompose the input tensor as  mps using SVD decomposition.

    Parameters
    ----------

    tensor : numpy array
        input tensor to be decomposed as MPS using SVD decomposition

    Returns
    _______

    mps : list
        list where each element is the rank-3 tensor as numpy array
        of the MPS QR decomposition of the input tensor
    """
    
    # Compute rank of the tensor
    rank = tensor.ndim
    logger.debug('rank: {}'.format(rank))
    # Dimensions of the legs
    leg_dim = tensor.shape
    logger.debug('leg_dim: {}'.format(leg_dim))
    m_step = copy.deepcopy(tensor)
    # eps value for numpy. For truncation
    eps = np.finfo(float).eps
    mps = []
    d_alphas = [1]
    ent_entropy = []
    # Reshape to matrix using first leg
    m_step = m_step.reshape(leg_dim[0], np.prod(leg_dim[1:]))
    size = 0.0
    for leg in range(rank-1):
        # loop over legs
        logger.debug('leg: {} m_step: {}'.format(leg, m_step.shape))
        # QR decomposition
        u_, s_, vh = np.linalg.svd(m_step, full_matrices=False)
        # Truncation by eps
        logger.debug('Before Truncation leg: {} u_: {}'.format(leg, u_.shape))
        logger.debug('Before Truncation leg: {} vh: {}'.format(leg, vh.shape))
        logger.debug('Before Truncation leg: {} s_: {}'.format(leg, s_.shape))
        u_ = u_[:, s_> eps]
        vh = vh[s_> eps, :]
        s_ = s_[s_> eps]
        logger.debug('leg: {} u_: {}'.format(leg, u_.shape))
        logger.debug('leg: {} vh: {}'.format(leg, vh.shape))
        logger.debug('leg: {} s_: {}'.format(leg, s_.shape))
        # Create Left Matrix
        left = np.diag(s_) @ vh
        d_alphas.append(left.shape[0])
        logger.debug('leg: {} left: {}'.format(leg, left.shape))
        # Compute entropy of the step
        s_step = - s_**2 @ np.log2(s_**2)
        ent_entropy.append(s_step)
        logger.info('leg : {} Entanglement Entropy: {}'.format(leg, s_step))
        # Convert q isometric matrix to rank-3 tensor
        # -u-s-vh- -> -u-s-vh-
        #              |
        u_ = u_.reshape(-1, leg_dim[leg], left.shape[0])
        logger.debug('leg: {} u_ tensor: {}'.format(leg, u_.shape))
        mps.append(u_)
        # Partition of following leg
        m_step = left.reshape(left.shape[0] * leg_dim[leg+1], -1)
        size = size + u_.size
    # Last leg procces
    m_step = m_step.reshape(-1, leg_dim[-1], 1)
    size = size + m_step.size
    logger.debug('leg: {} q_step tensor: {}'.format(rank-1, m_step.shape))
    mps.append(m_step)
    return mps, d_alphas, size, ent_entropy

def compose_mps(mps):
    """
    Given an input MPS it computes the final Tensor

    Parameters
    ----------

    mps : list
        list where each element is a rank-3 tensor that conforms de MPS

    Returns
    _______

    tensor : np.array
        numpy array with the correspondient tensor

    """
    tensor = mps[0]
    for i in range(1, len(mps)):
        tensor = contract_indices(
            tensor, mps[i], [tensor.ndim-1], [0])
    return tensor
