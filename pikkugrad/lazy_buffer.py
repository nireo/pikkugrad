from pikkugrad.dtype_op import LoadOp, UnaryOp, BinaryOp, TernaryOp, ReduceOp, dtypes
import numpy as np


class LazyBuffer:
    device = "CPU"
    dtype = dtypes.float32
    realized = None

    def __init__(self, buf):
        self._np = buf

    @property
    def shape(self):
        return self._np.shape

    def realize(x):
        return x

    @staticmethod
    def fromCPU(x):
        return LazyBuffer(x)

    def toCPU(self):
        return self._np

    @staticmethod
    def loadop(op, shape, arg=None):
        if op == LoadOp.RAND:
            return LazyBuffer(
                np.random.default_rng(arg).random(size=shape, dtype=np.float32)
            )
        elif op == LoadOp.CONST:
            return LazyBuffer(np.full(shape, arg))
        else:
            raise NotImplementedError(op)

    def contiguous(x):
        return x

    def const(self, x):
        return LazyBuffer(np.full_like(self._np, x))

    def e(self, op, *srcs):
        if op == UnaryOp.NEG:
            return LazyBuffer(-self._np)
        elif op == UnaryOp.EXP2:
            return LazyBuffer(np.exp2(self._np))
        elif op == UnaryOp.LOG2:
            return LazyBuffer(np.log2(self._np))
        elif op == UnaryOp.SIN:
            return LazyBuffer(np.sin(self._np))
        elif op == UnaryOp.SQRT:
            return LazyBuffer(np.sqrt(self._np))
        elif op == BinaryOp.ADD:
            return LazyBuffer(self._np + srcs[0]._np)
        elif op == BinaryOp.SUB:
            return LazyBuffer(self._np - srcs[0]._np)
        elif op == BinaryOp.MUL:
            return LazyBuffer(self._np * srcs[0]._np)
        elif op == BinaryOp.DIV:
            return LazyBuffer(self._np / srcs[0]._np)
        elif op == BinaryOp.MAX:
            return LazyBuffer(np.maximum(self._np, srcs[0]._np))
        elif op == BinaryOp.CMPLT:
            return LazyBuffer(self._np < srcs[0]._np)
        elif op == TernaryOp.WHERE:
            return LazyBuffer(np.where(self._np, srcs[0]._np, srcs[1]._np))
        else:
            raise NotImplementedError(op)

    def r(self, op, new_shape):
        assert len(self.shape) == len(
            new_shape
        ), "reduce shapes must have same dimensions"
        axis = tuple(i for i, (a, b) in enumerate(zip(self.shape, new_shape)) if a != b)
        if op == ReduceOp.SUM:
            return LazyBuffer(self._np.sum(axis, keepdims=True))
        elif op == ReduceOp.MAX:
            return LazyBuffer(self._np.max(axis, keepdims=True))
        else:
            raise NotImplementedError(op)

    # MovementOps
    def reshape(self, arg):
        return LazyBuffer(self._np.reshape(arg))

    def expand(self, arg):
        return LazyBuffer(np.broadcast_to(self._np, arg))

    def shrink(self, arg):
        return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])

    def permute(self, arg):
        return LazyBuffer(self._np.transpose(arg))

    def pad(self, arg):
        return LazyBuffer(np.pad(self._np, arg))

    def stride(self, arg):
        return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])
