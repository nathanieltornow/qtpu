ARG ARCH=x86_64

# Use cuQuantum appliance as base image
FROM nvcr.io/nvidia/cuquantum-appliance:24.03-${ARCH}

# cuquantums aer-simulator is very slow for mid-circuit measurments
RUN pip uninstall -y qiskit-aer

# Install dependencies
RUN pip install qiskit-aer "jax[cuda12]" numpy networkx quimb kahypar scipy optuna


RUN chown -R cuquantum:cuquantum /home/cuquantum/qtpu
RUN chmod -R g+w /home/cuquantum/qtpu



ENV PYTHONPATH "${PYTHONPATH}:/home/cuquantum/qtpu"
