"""
This module contains all functions needed for creating QLM
implementation for the ansatz of the Parent Hamiltonian paper:

    Fumiyoshi Kobayashi and Kosuke Mitarai and Keisuke Fujii
    Parent Hamiltonian as a benchmark problem for variational quantum
    eigensolvers
    Physical Review A, 105, 5, 2002
    https://doi.org/10.1103%2Fphysreva.105.052415

Authors: Gonzalo Ferro

"""

import pandas as pd
import numpy as  np
import qat.lang.AQASM as qlm
from qat.core import Result


def ansatz_qlm(nqubits, depth=3):
    """
    Implements QLM version of the Parent Hamiltonian Ansatz using
    parametric Circuit

    Parameters
    ----------

    nqubits: int
        number of qubits for the ansatz
    depth : int
        number of layers for the parametric circuit

    Returns
    _______

    qprog : QLM Program
        QLM program with the ansatz implementation in parametric format
    """

    qprog = qlm.Program()
    qbits = qprog.qalloc(nqubits)

    #Parameters for the PQC
    theta = [
        qprog.new_var(float, "\\theta_{}".format(i)) for i in range(2*depth)
    ]

    for d_ in range(0, 2*depth, 2):
        for i in range(nqubits):
            qprog.apply(qlm.RX(theta[d_]), qbits[i])
        for i in range(nqubits-1):
            qprog.apply(qlm.Z.ctrl(), qbits[i], qbits[i+1])
        qprog.apply(qlm.Z.ctrl(), qbits[nqubits-1], qbits[0])
        for i in range(nqubits):
            qprog.apply(qlm.RZ(theta[d_+1]), qbits[i])
    return qprog

def ansatz_qlm_general(nqubits, depth=3):
    """
    Implements QLM version of the Parent Hamiltonian Ansatz using
    parametric Circuit

    Parameters
    ----------

    nqubits: int
        number of qubits for the ansatz
    depth : int
        number of layers for the parametric circuit

    Returns
    _______

    qprog : QLM Program
        QLM program with the ansatz implementation in parametric format
    """

    qprog = qlm.Program()
    qbits = qprog.qalloc(nqubits)

    #Parameters for the PQC
    #theta = [
    #    qprog.new_var(float, "\\theta_{}".format(i)) for i in range(2*depth)
    #]
    
    theta = []
    indice = 0
    for d_ in range(0, 2*depth, 2):
        for i in range(nqubits):
            step = qprog.new_var(float, "\\theta_{}".format(indice))
            qprog.apply(qlm.RX(step), qbits[i])
            theta.append(step)
            indice = indice + 1
        for i in range(nqubits-1):
            qprog.apply(qlm.Z.ctrl(), qbits[i], qbits[i+1])
        qprog.apply(qlm.Z.ctrl(), qbits[nqubits-1], qbits[0])
        for i in range(nqubits):
            step = qprog.new_var(float, "\\theta_{}".format(indice))
            qprog.apply(qlm.RZ(step), qbits[i])
            indice = indice + 1
            theta.append(step)
    return qprog

def proccess_qresults(result, qubits, complete=True):
    """
    Post Process a QLM results for creating a pandas DataFrame

    Parameters
    ----------

    result : QLM results from a QLM qpu.
        returned object from a qpu submit
    qubits : int
        number of qubits
    complete : bool
        for return the complete basis state.
    """

    # Process the results
    if complete:
        states = []
        list_int = []
        list_int_lsb = []
        for i in range(2**qubits):
            reversed_i = int("{:0{width}b}".format(i, width=qubits)[::-1], 2)
            list_int.append(reversed_i)
            list_int_lsb.append(i)
            states.append("|" + bin(i)[2:].zfill(qubits) + ">")

        probability = np.zeros(2**qubits)
        amplitude = np.zeros(2**qubits, dtype=np.complex_)
        for samples in result:
            probability[samples.state.lsb_int] = samples.probability
            amplitude[samples.state.lsb_int] = samples.amplitude

        pdf = pd.DataFrame(
            {
                "States": states,
                "Int_lsb": list_int_lsb,
                "Probability": probability,
                "Amplitude": amplitude,
                "Int": list_int,
            }
        )
    else:
        list_for_results = []
        for sample in result:
            list_for_results.append([
                sample.state, sample.state.lsb_int, sample.probability,
                sample.amplitude, sample.state.int,
            ])

        pdf = pd.DataFrame(
            list_for_results,
            columns=['States', "Int_lsb", "Probability", "Amplitude", "Int"]
        )
        pdf.sort_values(["Int_lsb"], inplace=True)
    return pdf

def solving_ansatz(qlm_circuit, nqubit, qlm_qpu, reverse=True):
    """
    Solving a complete qlm circuit

    Parameters
    ----------

    qlm_circuit : QLM circuit
        qlm circuit to solve
    nqubit : int
        number of qubits of the input circuit
    qlm_qpu : QLM qpu
        QLM qpu for solving the circuit
    reverse : True
        This is for ordering the state from left to right
        If False the order will be form right to left

    Returns
    _______

    mps_state : numpy array
        State of the input circuit in nqubit-tensor format
    """
    # Creating the qlm_job
    job = qlm_circuit.to_job()
    qlm_state = qlm_qpu.submit(job)
    if not isinstance(qlm_state, Result):
        qlm_state = qlm_state.join()
        # time_q_run = float(result.meta_data["simulation_time"])

    pdf_state = proccess_qresults(qlm_state, nqubit, True)
    # For keep the correct qubit order convention for following
    # computations
    if reverse:
        pdf_state.sort_values('Int', inplace=True)
    # A n-qubit-tensor is prefered for returning
    state = np.array(pdf_state['Amplitude'])
    mps_state = state.reshape(tuple(2 for i in range(nqubit)))
    return mps_state
