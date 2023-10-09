from typing import ClassVar


# Tensor represents a multi-dimensional scalar array.
class Tensor:
    no_grad: ClassVar[bool] = False
