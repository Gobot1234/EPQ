from __future__ import annotations

import math
from typing import NamedTuple, Protocol


class Position(NamedTuple):
    x: float
    y: float
    z: float

    @classmethod
    def origin(cls) -> Position:
        return cls(0, 0, 0)

    def __add__(self, other: Position) -> Position:
        """Find the sum of two vectors."""
        return self.__class__(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Position) -> Position:
        """Find the difference between two vectors."""
        return self.__class__(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __or__(self, other: Position) -> float:
        """The magnitude of the vector from self to other.

        Equivalent to:
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
    position: Position | classmethod[property[Position]]
