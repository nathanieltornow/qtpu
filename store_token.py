import sys

from qiskit.providers.ibmq import IBMQ


token = sys.argv[1]

IBMQ.save_account(
    token=token,
    # hub="ibm-q-research-2",
    # group="tu-munich-1",
    # project="main",
    overwrite=True,
)

IBMQ.load_account()

print(IBMQ.providers())