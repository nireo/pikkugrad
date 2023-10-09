from typing import ClassVar


class Func:
    def __init__(self, device: str, *tensors: Tensor) -> None:
        self.device = device


# Tensor represents a multi-dimensional scalar array.
class Tensor:
    no_grad: ClassVar[bool] = False
