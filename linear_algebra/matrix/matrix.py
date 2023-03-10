from ..vector import Vector
from typing import List, Callable
from functools import partialmethod, wraps

__all__ = ["Matrix"]


def det_cache(f):
    results = {}

    @wraps(f)
    def wrapper(self: "Matrix"):
        flat = tuple(tuple(vector.scalars) for vector in self.rows)
        if (result := results.get(flat)) is None:
            results[flat] = result = f(self)
        return result
    return wrapper


class Matrix:
    def __init__(self, *rows: Vector):
        self.rows = list(rows)
        self.columns = self.transpose().rows

    @classmethod
    def __init_without_transpose(cls, rows: List[Vector], columns: List[Vector]) -> "Matrix":
        matrix = cls.__new__(cls)
        matrix.rows = rows
        matrix.columns = columns
        return matrix

    @property
    def size(self):
        return len(self.rows), len(self.columns)

    def transpose(self) -> "Matrix":
        new_matrix = [Vector() for _ in self.rows[0].scalars]
        for row in self.rows:
            for s, scalar in enumerate(row.scalars):
                new_matrix[s].scalars.append(scalar)
        return Matrix.__init_without_transpose(new_matrix, self.rows)

    def _mul(self, other):
        if isinstance(other, (float, int)):
            return Matrix(*[Vector(*[other * scalar for scalar in row.scalars]) for row in self.rows])
        assert self.size[1] == other.size[0], "Orders of matrices do not match"
        matrix = []
        transposed_other = other.transpose()
        for i in range(len(self.rows)):
            vector = Vector()
            for j in range(len(transposed_other.rows)):
                vector.scalars.append(self.rows[i] * transposed_other.rows[j])
            matrix.append(vector)
        return Matrix(*matrix)

    @classmethod
    def from_function(cls, n_rows: int, n_columns: int, function: Callable[[int, int], int]) -> "Matrix":
        return cls(*[Vector(*[function(i, j) for i in range(n_columns)]) for j in range(n_rows)])

    zeros: Callable[[int, int], "Matrix"] = partialmethod(from_function, function=lambda i, j: 0)
    ones: Callable[[int, int], "Matrix"] = partialmethod(from_function, function=lambda i, j: 1)

    @classmethod
    def unit(cls, size: int):
        return cls.from_function(size, size, lambda i, j: int(i == j))

    def remove_row(self, row):
        return Matrix(*(self.rows[:row] + self.rows[row + 1:]))

    def remove_column(self, column):
        return Matrix.remove_row(self.transpose(), column).transpose()

    def assert_square(self):
        assert self.size[0] == self.size[1], "The matrix must be square"

    def trace(self):
        self.assert_square()
        return sum([self.rows[i].scalars[i] for i in range(len(self.rows[0].scalars))])

    def __mul__(self, other):
        return Matrix._mul(self, other)

    def __rmul__(self, other):
        return Matrix._mul(other, self)

    @det_cache
    def __abs__(self) -> int:
        self.assert_square()
        start = 0
        result = 0
        if self.size[0] == 1:
            return self.rows[0].scalars[0]

        for j in range(self.size[0]):
            result += ((-1) ** (start + j)) * \
                      self.rows[start].scalars[j] \
                      * abs(self.remove_row(start).remove_column(j))

        return result

    def __pow__(self, power, modulo=None):
        result = self
        for _ in range(power - 1):
            result *= self
        return result

    def __str__(self):
        return "\n".join(str(row) for row in self.rows)

    det: Callable = __abs__
