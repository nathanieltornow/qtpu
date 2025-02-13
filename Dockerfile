ARG ARCH=x86_64

# Use cuQuantum appliance as base image
FROM nvcr.io/nvidia/cuquantum-appliance:24.03-${ARCH}

# cuquantums aer-simulator is very slow for mid-circuit measurments
# RUN pip uninstall -y qiskit-aer

# Install dependencies
# RUN pip install qiskit-aer "jax[cuda12]" numpy networkx quimb kahypar scipy optuna
RUN pip install numpy networkx quimb kahypar scipy optuna


ENV PYTHONPATH "${PYTHONPATH}:/home/cuquantum/qtpu"
