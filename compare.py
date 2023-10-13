

import sys
import numpy as np
from ansatz_mps import ansatz, get_angles
from parent_hamiltonian_mps import PH_MPS

sys.path.append("../WP3_Benchmark/tnbs/BTC_04_PH/PH")
from ansatzes import run_ansatz
from parent_hamiltonian import PH


def do_mps(nqubits, depth):
    # MPS uisng My code
    angles = get_angles(depth)
    mps = ansatz(nqubits, depth, angles)
    ph_conf = {"save": False}
    ph_ob_mps = PH_MPS(mps, True, **ph_conf)
    ph_ob_mps.local_ph()
    return ph_ob_mps.pauli_pdf,  angles

def do_myqlm(nqubits, depth):
    conf = {
        "nqubits": nqubits,
        "depth": depth,
        "ansatz" : "simple01",
        "qpu_ansatz": "c",
        "t_inv": True,
        "folder":  "/EASQC/",
        "save": False,
        "solve" : True,
        "submit": False
    }
    solved_ansatz = run_ansatz(**conf)
    angles = solved_ansatz["parameters"]
    pdf = solved_ansatz["state"]
    ph_conf = {"save": False}
    ph_ob = PH(pdf[["Amplitude"]], True, **ph_conf)
    ph_ob.local_ph()
    return ph_ob.pauli_pdf, angles


if __name__ == "__main__":
    import time
    import logging
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        #level=logging.INFO
        level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')
    # Given a state Compute its Parent Hamiltonian
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nqubits",
        dest="nqubits",
        type=int,
        help="Number of qbits for the ansatz.",
        default=None,
    )
    parser.add_argument(
        "-depth",
        dest="depth",
        type=int,
        help="Depth for ansatz.",
        default=None,
    )
    parser.add_argument(
        "--mps",
        dest="mps",
        default=False,
        action="store_true",
        help="for mps",
    )
    parser.add_argument(
        "--myqlm",
        dest="myqlm",
        default=False,
        action="store_true",
        help="for myqlm",
    )
    args = parser.parse_args()
    nqubits = args.nqubits
    depth = args.depth
    if args.mps:
        tick = time.time()
        pdf_mps, angles_mps = do_mps(nqubits, depth)
        tack = time.time()
        mps_time = tack - tick
        print("mps_time: {}".format(mps_time))
        mps_angles = []
        for angle in angles_mps:
            mps_angles = mps_angles + angle
    if args.myqlm:
        tick = time.time()
        pdf_myqlm, angles_myqlm = do_myqlm(nqubits, depth)
        tack = time.time()
        myqlm_time = tack - tick
        print("myqlm_time: {}".format(myqlm_time))
        myqlm_angles = list(angles_myqlm["value"])
        
    if (args.mps) and (args.myqlm):
        test_angles = np.isclose(np.array(mps_angles), np.array(myqlm_angles)).all()
        assert test_angles
        Test = np.isclose(
            pdf_mps["PauliCoefficients"].astype(float),
            pdf_myqlm["PauliCoefficients"].astype(float),
        ).all()
        print("Pauli Coefficients testing: {}".format(Test))
        if Test == False:
            error = pdf_mps["PauliCoefficients"].astype(float) - pdf_myqlm["PauliCoefficients"].astype(float)
            error = np.sqrt((error **2).sum())
            print("BE AWARE Paulti Coefs are not same. Error: {}".format(error))


