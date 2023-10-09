from typing import Final, NamedTuple, Optional
from enum import Enum, auto
import numpy as np


# DType holds numpy information about a given type and also prints it for debugging.
class DType(NamedTuple):
    name: str
    np: Optional[type]  # numpy type
    sz: int = 1

    def __repr__(self) -> str:
        return f"dtypes.{self.name}"


# dtypes represents all of the possible different type values.
# represents all of different tensor values a tensor can contain.
class dtypes:
    @staticmethod
    def is_float(x: DType) -> bool:
        return x == dtypes.float32

    float32: Final[DType] = DType("float32", np.float32)
    int32: Final[DType] = DType("int32", np.int32)
    bool: Final[DType] = DType("bool", np.bool_)


# UnaryOp represents different unary ops, like: -a This also includes
# some function calls like: log_2(x), sin(x), exp_2(x)
class UnaryOp(Enum):
    NOOP = auto()
    EXP2 = auto()
    LOG2 = auto()
    CAST = auto()
    SIN = auto()
    NEG = auto()
    SQRT = auto()


# BinaryOp represents some operations done on two operands.
# For example: max(a, b), a + b, a - b
class BinaryOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    MAX = auto()
    MOD = auto()
    DIV = auto()
    CMPLT = auto()  # compare less than


# Different reduce operations, e.g. finding max or summing
class ReduceOp(Enum):
    SUM = auto()
    MAX = auto()


# TernaryOp contains operations to find elements based on a condition.
class TernaryOp(Enum):
    WHERE = auto()


# MovementOp contains different operations to manipulate data.
class MovementOp(Enum):
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SHRINK = auto()
    STRIDE = auto()


# LoadOp contains different operations of loading data.
class LoadOp(Enum):
    EMPTY = auto()
    RAND = auto()
    CONST = auto()
    FROM = auto()
    CONTIGUOUS = auto()
    CUSTOM = auto()
