from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .asteroid import Asteroid
from .station import Body, Planet, Station

if TYPE_CHECKING:
    from .position import HasPosition, Position


@dataclass(slots=True)
class Miner:
    base_station: Planet
    position: Position = ...
    carrying: Asteroid | None = None

    money_made: float = 0
    fuel: float = 30_000  # the mass of fuel the miner has in kg.
    efficiency: float = 500_000_000  # conversion rate of the 1kg of fuel to energy in Joules. This might have to be a prop depedning on our location

    def __post_init__(self) -> None:
        self.position = self.base_station.position

    @property
    def profit(self) -> float:
        FUEL_COST = 0.4
        return self.money_made - (30_000 * FUEL_COST)

    @property
    def base_mass(self) -> float:
        return 50_000

    @property
    def mass(self) -> float:
        asteroid_mass = 0 if self.carrying is None else self.carrying.mass
        return self.base_mass + self.fuel + asteroid_mass

    @property
    def distance_to_home(self) -> float:
        return self.position | self.base_station.position

    def fuel_to_get_to(self, target: Body) -> float:
        delta_v = (self.carrying or self.base_station).delta_v_for(self)
        energy = 1 / 2 * self.mass * delta_v ** 2  # constant acceleration means this is all the energy we need
        # acceleration = G.value * self.mass * target.mass / (target.position | self.position ** 2)
        return energy / self.efficiency

    def travel_to(self, target: HasPosition) -> None:
        if isinstance(target, Asteroid):
            self.fuel -= self.fuel_to_get_to(target)
            self.carrying = target
        elif isinstance(target, Station):
            if self.carrying is not None:
                self.money_made += self.carrying.price
                self.carrying = None
            self.fuel -= self.fuel_to_get_to(target)

        position = target.position

        self.position = position  # type: ignore
