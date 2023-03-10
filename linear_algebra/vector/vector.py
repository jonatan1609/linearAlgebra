from fractions import Fraction

__all__ = ["Vector"]


class Vector:
    def __init__(self, *scalars: int):
        self.scalars = list(scalars)

    def norm(self):
        return Fraction(*((self * self) ** 0.5).as_integer_ratio())

    def __add__(self, other: "Vector"):
        return Vector(*[scalar + other for scalar, other in zip(self.scalars, other.scalars)])

    def __sub__(self, other):
        return self + -1 * other

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector(*[scalar * other for scalar in self.scalars])
        assert len(other.scalars) == len(self.scalars), "both vectors must be of the same size."
        return sum(self.scalars[i] * other.scalars[i] for i in range(len(self.scalars)))

    __rmul__ = __mul__

    def __str__(self):
        return " ".join(str(s) for s in self.scalars)
