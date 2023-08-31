"""
Solution for T01_2 from TN Korean course
"""

import numpy as np
import itertools as it
import functools as ft


def creator_l(l, dim):
    """
    Creates a creation operator for the inputs:

    Parameters
    ----------
    l : int
        Position where the particle will be created
    dim : int
        Dimension of the operator

    Returns
    _______

    creator_l : np array
        Numpy array with the desired creator operator
    """

    if l >= dim:
        raise ValueError("Position higher than dimension")

    identity = np.identity(2)
    make_list = [identity for i in range(dim)]
    # 1 qbit creation operator
    c_plus = np.zeros((2,2))
    c_plus[1,0] = 1
    make_list[l] = c_plus
    c_plus_l = ft.reduce(np.kron, make_list)
    return c_plus_l


def tbc_qbit_image(t_vector):
    """
    Creates the Hamiltonian for a tight binding chain of L elements.
    Dimension will be L-1

    Parameters
    ----------
    ground state energy : float
        Energy of the ground state
    eigvalues : np.array
        Array with the eigenvalues of the Hamiltonian
    eigvectors : np.array
       Array with eigenvectors of the Hamiltonian
    h_tb : np.array
        Array with the Hamiltonian of the Chain
    d_g : int
        Degeneracy of Ground State
    """

    dim =len(t_vector) + 1
    first = sum([t_l * creator_l(l+1, dim) @ creator_l(l, dim).T \
        for l, t_l in enumerate(t_vector)])
    # Compute Hamiltonian
    h_tb = -first - np.conjugate(first).T
    # Compute Eigenvalues and Eigenvectors
    eigvalues, eigvectors = np.linalg.eig(h_tb)
    eigvalues = np.real(eigvalues)
    # Compute Degeneracy of Ground State
    d_g = np.isclose(eigvalues, np.min(eigvalues)).sum()
    return np.min(eigvalues), eigvalues, eigvectors, h_tb,  d_g

def tight_bind_matlab(t_):
    #creates the hamiltonian
    h_1p = np.diag(-t_,-1)
    h_1p = h_1p + np.conjugate(h_1p).T # Hermitianize
    eigvalues, eigvectors = np.linalg.eig(h_1p)
    e_g = np.sum(eigvalues[eigvalues<0])
    d_g = 2**(sum(np.isclose(eigvalues, 0)))
    return e_g, eigvalues, eigvectors, h_1p, d_g


def get_vector(l, dim):
    """
    Creates a vector do input dimension dim. All elements will be zero
    but the l that will be one

    Parameters
    ----------
    l : int
        Position where the particle will be created
    dim : int
        Dimension of the operator

    Returns
    _______

    creator_l : np array
        Numpy array with the desired vector
    """
    v = np.zeros((dim, 1))
    v[l] = 1
    return v

def tb_chain_2(t_vector):
    """
    Creates the Hamiltonian for a tight binding chain of L elements.
    Dimension will be L-1

    Parameters
    ----------
    t_vector : numpy array
        Vector with the iteration elements of the chain
    """
    # Creates the |l+1><l| operators
    h1p = sum([t* get_vector(l+1, len(t_vector) + 1) \
        @ get_vector(l, len(t_vector) + 1).T for l, t in enumerate(t_vector)])
    # Adds the |l+1><l|.T operators
    h1p = -h1p - np.conjugate(h1p).T
    # Computes Eigenvalues
    eigvs = np.linalg.eigvals(h1p)
    eigvs.sort()
    # Energy of ground state will be the summ of all the negative eigenv
    egs = sum(eigvs[eigvs < 0])
    dg = 2**np.isclose(eigvs, 0).sum()
    return egs, dg, eigvs

def tb_chain_old(t_vector):
    """
    Creates the Hamiltonian for a tight binding chain of L elements.
    Dimension will be L-1

    Parameters
    ----------
    t_vector : numpy array
        Vector with the iteration elements of the chain
    """

    dim =len(t_vector) + 1
    h_tb = np.zeros((2**dim, 2**dim))
    for l, t_l in enumerate(t_vector):
        c_l_1 = creator_l(l+1, dim)
        a_l = creator_l(l, dim).T
        c_l = creator_l(l, dim)
        a_l_1  = creator_l(l+1, dim).T
        h_tb = h_tb + t_l * c_l_1 @a_l + np.conjugate(t_l) * c_l @ a_l_1
    eigvs = np.linalg.eigvals(h_tb)
    eigvs.sort()
    egs = eigvs[0]
    dg = np.isclose(eigvs, egs).sum()
    return egs, dg, eigvs, h_tb
        
