ARG ARCH=x86_64

# Use cuQuantum appliance as base image
FROM nvcr.io/nvidia/cuquantum-appliance:24.03-${ARCH}


# uninstall overriden qiskit and qiskit-aer
# RUN pip uninstall -y qiskit qiskit-aer

# RUN pip install "jax[cuda12]"





RUN pip install "jax[cuda12]" numpy networkx quimb kahypar scipy optuna


WORKDIR /workspace

COPY . .

WORKDIR /workspace/examples





# ENV UCX_TLS=cma,cuda,cuda_copy,cuda_ipc,mm,posix,self,shm,sm,sysv,tcp


