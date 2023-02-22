from ..vector import Vector

__all__ = ["Matrix"]


class Matrix:
    def __init__(self, *rows: Vector):
        self.rows = list(rows)
        self.columns = self.transpose().rows

    def transpose(self) -> "Matrix":
        matrix = [Vector() for _ in range(len(self.rows[0].scalars))]
        for row in self.rows:
            for s, scalar in enumerate(row.scalars):
                matrix[s].scalars.append(scalar)

        return Matrix(*matrix)

    def _mul(self, other):
        if isinstance(other, (float, int)):
            return Matrix(*[Vector(*[other * scalar for scalar in row.scalars]) for row in self.rows])
        assert len(self.rows[0].scalars) == len(other.transpose().rows[0].scalars), "Orders of matrices does not match"
        matrix = []
        transposed_other = other.transpose()
        for i in range(len(self.rows)):
            vector = Vector()
            for j in range(len(transposed_other.rows)):
                vector.scalars.append(self.rows[i] * transposed_other.rows[j])
            matrix.append(vector)
        return Matrix(*matrix)

    def remove_row(self, row):
        return Matrix(*(self.rows[:row] + self.rows[row + 1:]))

    def remove_column(self, column):
        return Matrix.remove_row(self.transpose(), column).transpose()

    def assert_square(self):
        assert len(self.transpose().rows[0].scalars) == len(self.rows[0].scalars), "The matrix must be square"

    def trace(self):
        self.assert_square()
        return sum([self.rows[i].scalars[i] for i in range(len(self.rows[0].scalars))])

    def __mul__(self, other):
        return Matrix._mul(self, other)

    def __rmul__(self, other):
        return Matrix._mul(other, self)

    def __abs__(self):
        self.assert_square()
        start = 0
        result = 0
        size = len(self.rows[0].scalars)

        if size == 2:
            return self.rows[0].scalars[0] * self.rows[1].scalars[1] \
                 - self.rows[0].scalars[1] * self.rows[1].scalars[0]

        for j in range(size):
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

    det = __abs__
