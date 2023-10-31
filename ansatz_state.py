import pandas as pd
import numpy as np
from  utils_ph import create_folder
from ansatz_mps import ansatz, get_angles
from tensornetworks import compose_mps, contract_indices_one_tensor


def state_computation(**configuration):
    """
    Computes Parent Hamiltonian for an ansatz
    """
    # Ansatz configuration
    nqubits = configuration["nqubits"]
    depth = configuration["depth"]
    truncate = configuration["truncate"]
    # PH configuration
    mpstest = configuration["mpstest"]
    save = configuration["save"]
    folder_name = configuration["folder"]

    folder_name = create_folder(folder_name)
    base_fn = folder_name + "nqubits_{}_depth_{}".format(
        str(nqubits).zfill(2), depth)
    # Build Angles
    angles = get_angles(depth)
    list_angles = []
    for angle in angles:
        list_angles = list_angles + angle
    param_names = ["\\theta_{}".format(i) for i, _ in enumerate(list_angles)]
    pdf_angles = pd.DataFrame([param_names, list_angles], index=["key", "value"]).T
    if save:
        pdf_angles.to_csv(base_fn + "_parameters.csv", sep=";")
    # Build MPS of the ansatz
    mps = ansatz(nqubits, depth, angles, truncate)
    tensor = compose_mps(mps)
    print(tensor.shape)
    print(tensor.ndim)
    state = contract_indices_one_tensor(tensor, [(0, tensor.ndim-1)])
    state = state.reshape(np.prod(state.shape))
    print(state)
if __name__ == "__main__":
    import logging
    #logging.basicConfig(
    #    format='%(asctime)s-%(levelname)s: %(message)s',
    #    datefmt='%m/%d/%Y %I:%M:%S %p',
    #    level=logging.INFO
    #    #level=logging.DEBUG
    #)
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
        "--truncate",
        dest="truncate",
        default=False,
        action="store_true",
        help="Truncating the SVD. Float resolution will be used",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    parser.add_argument(
        "--mpstest",
        dest="mpstest",
        default=False,
        action="store_true",
        help="New Contractions Tested",
    )
    parser.add_argument(
        "-folder",
        dest="folder",
        type=str,
        default="",
        help="Folder for Storing Results",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the AE algorihtm configuration."
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()
    if args.print:
        print(args)
    if args.execution:
        state_computation(**vars(args))

