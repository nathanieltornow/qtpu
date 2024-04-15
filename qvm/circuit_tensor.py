from numpy.typing import NDArray


class ParamTensor:
    def __init__(self, params: NDArray) -> None:
        self._params = params
        # execute every indexcombination
