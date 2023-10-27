"""
Routines for simulating quantum circuits using tensor networks
"""
import logging
import numpy as np
import pandas as pd
from scipy.linalg import svd
from tensornetworks import contract_indices
logger = logging.getLogger('__name__')

def my_svd(array, truncate=False, t_v=None):
    """
    Execute SVD.

    Parameters
    ----------
    array : np.array

        Arry with matrix for SVD
    truncate : Bool
        For truncating SVD. If t_v is None then the truncation will be
        using float prcision of the system
    t_v : float
        In truncate is True then the t_v float will use as truncation
        threshold

    Returns
    -------

    u_, s, vh : np.arrays
        numpy arrays with the SVD of the input array
    """
    u_, s_, vh = svd(array, full_matrices=False)
    logger.debug('Before Truncation u_: {}'.format(u_.shape))
    logger.debug('Before Truncation vh: {}'.format(vh.shape))
    logger.debug('Before Truncation s_: {}'.format(s_.shape))
    if truncate == True:
        # For truncate SVD
        logger.debug('Truncation s_: {}'.format(s_))
        if t_v is None:
            # If not Truncation limit we use minimum float precision
            eps = np.finfo(float).eps
            u_ = u_[:, s_> eps]
            vh = vh[s_> eps, :]
            s_ = s_[s_> eps]
        else:
            u_ = u_[:, s_> t_v]
            vh = vh[s_> t_v, :]
            s_ = s_[s_> t_v]
        logger.debug('After Truncation u_: {}'.format(u_.shape))
        logger.debug('After Truncation vh: {}'.format(vh.shape))
        logger.debug('After Truncation s_: {}'.format(s_.shape))
    return u_, s_, vh

def bitfield(n_int: int, size: int):
    """Transforms an int n_int to the corresponding bitfield of size size

    Parameters
    ----------
    n_int : int
        integer from which we want to obtain the bitfield
    size : int
        size of the bitfield

    Returns
    ----------
    full : list of ints
        bitfield representation of n_int with size size

    """
    aux = [1 if digit == "1" else 0 for digit in bin(n_int)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size - right.size, 0))
    full = np.concatenate((left, right))
    return full.astype(int)

def bitfield_to_int(lista):
    """Transforms the bitfield list to the corresponding int
    Parameters
    ----------
    lista : ist of ints
        bitfield

    Returns
    ----------
    integer : int
        integer obtained from it's binary representation.
    """

    integer = 0
    for i in range(len(lista)):
        integer += lista[-i - 1] * 2**i
    return int(integer)


def compose_mps(mps, last=True):
    """
    Given an input MPS it computes the final Tensor
    The rank-3 tensor MUST HAVE following indexing:

                        0-o-2
                          |
                          1
    Where 1 is the physical leg.
    The computation done is:
                    _____________
                    |           |
                    -o-o-o-...-o-
                     | | |     |
    Or
                    o-o-o-...-o
                    | | |     |

    Parameters
    ----------

    mps : list
        list where each element is a rank-3 tensor that conforms de MPS

    Returns
    _______

    tensor : np.array
        numpy array with the correspondient tensor

    """

    if all([mps_.ndim == 3 for mps_ in mps[1:-1]]) == False:
        raise ValueError("All tensors except first and last MUST BE of rank-3")

    start_tensor = mps[0]
    end_tensor = mps[-1]

    if (start_tensor.ndim == 3) and (end_tensor.ndim == 3):
        # All tensors are rank-3 tensors even first and last one
        tensor = mps[0]
        #Train contraction
        for i in range(1, len(mps)-1):
            tensor = contract_indices(tensor, mps[i], [tensor.ndim-1], [0])
        # ----Tensor n-1----  Tensor n
        # -o-o-o-...-o- - -o-
        # || | |     |     ||
        # |_________________|
        if last:
            tensor = contract_indices(
                tensor, mps[-1], [tensor.ndim-1, 0], [0, 2])
        else:
            tensor = contract_indices(
                tensor, mps[-1], [tensor.ndim-1], [0])
    elif (start_tensor.ndim == 2) and (end_tensor.ndim == 2):
        # First and Last tensor are rank-2 tensors
        tensor = mps[0]
        for i in range(1, len(mps)):
            tensor = contract_indices(tensor, mps[i], [tensor.ndim-1], [0])
    else:
        raise ValueError("First and Last tensor MUST BE rank-2 or ran-3 tensors")
    return tensor

def get_state_from_mps(mps):
    """
    From a MPS computes the final state as a pandas DataFrame

    Parameters
    ----------

    mps : list
        Each element is a rank-3 tensor from a MPS
    """
    nqubits = len(mps)
    state_list = [bitfield(i, nqubits) for i in range(2**nqubits)]
    int_lsb = [bitfield_to_int(list(np.flip(a_i)))  for a_i in state_list]
    state_list = [np.flip(s) for s in state_list]
    pdf = pd.DataFrame(
        [state_list, int_lsb, range(2**nqubits)],
        index=["States", "Int", "Int_lsb"]).T
    final_tensor = compose_mps(mps)
    final_state = final_tensor.reshape(2**nqubits)
    pdf["Amplitude"] = final_state
    return pdf


def apply_local_gate(mps, gates):
    """
    Apply  local gates on several rank 3-tensors.
    The rank-3 tensor MUST HAVE following indexing:

                        0-o-2
                          |
                          1

    Where 1 is the physical leg.
    The computation done is:
                    -o- -o- ... -o-
                     |   |       |
                     o   o       o
                     |   |       |
    Parameters
    ----------

    mps : list
        Each element is a rank-3 tensor from a MPS
    gates : list
        Each element is a local gate

    Returns
    _______

    o_qubits : list
        Each element is the resulting rank-3 tensor
    """
    o_qubits = []
    for q_, g_ in zip(mps, gates):
        o_qubits.append(contract_indices(
            q_, g_, [1], [0]).transpose(0, 2, 1))
    return o_qubits

def apply_2qubit_gate(tensor1, tensor2, gate = None, truncate=False, t_v=None):
    """
    Executes a 2-qubit gate between 2 rank-3 tensors
    The rank-3 tensors MUST HAVE following indexing:

                        0-o-2
                          |
                          1

    Where 1 is the physical leg.
    Following Computation is done:

        -o-o-
         | |  -> -o- -> SVD -> -o- -o-
         -o-                    |   |

    """
    logger.debug("tensor1: %s", tensor1.shape)
    logger.debug("tensor2: %s", tensor2.shape)
    d_physical_leg = tensor1.shape[1]

    if gate is None:
        return tensor1, tensor2
    else:
        # 0-o-o-3
        #   | |
        #   1 2
        step = contract_indices(tensor1, tensor2, [2], [0])
        # Reshape 2 physical legs to 1
        # 0-o-o-3     -o-
        #   | |   ->   |
        #   1 2      d1*d2
        step = step.reshape((tensor1.shape[0], -1, tensor2.shape[2]))
        logger.debug("Tensor1-Tensor2: %s", step.shape)
        # Contraction Tensor1-2 with Gate
        # 0-o-2    0-o-1    0-o-2
        #  1|0  ->   |   ->   |
        #   o        2        1
        #   |1
        step = contract_indices(step, gate, [1], [0])
        step = step.transpose(0,2,1)
        logger.debug("Tensor1-Gate-Tensor2: %s", step.shape)

        # SVD
        # Preparing SVD
        # -o- -> dim1 -o- dim2
        #  |
        dim1 = tensor1.shape[0] * tensor1.shape[1]
        dim2 = tensor2.shape[1] * tensor2.shape[2]
        step = step.reshape(dim1, dim2)

        logger.debug("Matrix for SVD : %s", step.shape)
        u_, s_, vh = my_svd(step, truncate, t_v)
        logger.debug("u_: %s", u_.shape)
        logger.debug("s_: %s", s_.shape)
        logger.debug("vh: %s", vh.shape)

        # Obtaining 2 new rank-3 tensors
        #dim1 -o- dim2 -> -o- -o-
        #                  |   |

        # Reshaping u_ as left tensor
        new_shape = [0, 0, 0]
        new_shape[1] = d_physical_leg
        new_shape[0] = tensor1.shape[0]
        new_shape[2] = -1
        #left = u_ @ np.diag(s_)
        left =  u_
        left = left.reshape(tuple(new_shape))
        logger.debug("Left Tensor:, %s", left.shape)

        #logger.debug("Right Tensor:, %s", right.shape)
        new_shape = [0, 0, 0]
        new_shape[1] = d_physical_leg
        new_shape[0] = -1
        new_shape[2] = tensor2.shape[2]
        # Reshaping other part as right tensor
        right =  np.diag(s_) @ vh
        # right = vh
        right = right.reshape(tuple(new_shape))
        logger.debug("Right Tensor:, %s", right.shape)

        return left, right
