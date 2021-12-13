from __future__ import annotations

import math
from datetime import datetime
from typing import NamedTuple, Protocol

from typing_extensions import Self


class Position(NamedTuple):
    """A cartesian 3D vector."""

    x: float
    y: float
    z: float

    @classmethod
    def origin(cls) -> Self:
        return cls(0, 0, 0)

    def __add__(self, other: Position) -> Self:
        """Find the sum of two vectors."""
        return self.__class__(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Position) -> Self:
        """Find the difference between two vectors."""
        return self.__class__(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, n: int) -> Self:
        """Multiply a vector by a factor n."""
        return self.__class__(
            self.x * n,
            self.y * n,
            self.z * n,
        )

    def __matmul__(self, other: Position) -> float:
        """The dot product of the two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __or__(self, other: Position) -> float:
        """The magnitude of the vector from self to other.

        Equivalent to (but faster):
        ```py
        (
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2 +
        )  ** 0.5
        ```
        """
        return math.dist(self, other)


class HasPosition(Protocol):
    @property
    def position(self) -> Position:
        ...

    def position_at(self, dt: datetime) -> Position:
        ...
