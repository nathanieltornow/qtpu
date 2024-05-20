# Use cuQuantum appliance as base image
FROM nvcr.io/nvidia/cuquantum-appliance:23.10


# uninstall overriden qiskit and qiskit-aer
RUN pip uninstall -y qiskit qiskit-aer

RUN pip install "qiskit<1.0.0" qiskit-aer-gpu "jax[cuda12]" numpy networkx quimb kahypar scipy optuna

ENV UCX_TLS=cma,cuda,cuda_copy,cuda_ipc,mm,posix,self,shm,sm,sysv,tcp
