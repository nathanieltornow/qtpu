# QVM - Quantum Virtual Machine

QVM is a framework for the scalable execution of quantum circuits using both quantum and classical accelerators (namely QPUs and GPUs).


## Development

### Using Docker

Build the image with all its dependencies:
```sh
docker build -t qvm .
```

Run the docker container while mounting the project's directory into `/home/cuquantum/qvm`:
```sh
docker run --gpus all -it --rm -v $(pwd):/home/cuquantum/qvm qvm
```