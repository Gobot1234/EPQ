from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from sympy.solvers import solve
from sympy import Symbol, log


from .asteroid import Asteroid, Contents
from .station import Body, Planet

if TYPE_CHECKING:
    from .position import HasPosition, Position


@dataclass(slots=True)
class Miner:
    base_station: Planet
    position: Position = ...
    carrying: list[Contents] | None = None
    asteroid: Asteroid = None

    money_made: float = 0
    fuel: float = 30_000  # the mass of fuel the miner has in kg.
    efficiency: float = 500_000_000  # conversion rate of the 1kg of fuel to energy in Joules. This might have to be a prop depedning on our location

    def __post_init__(self) -> None:
        self.position = self.base_station.position

    @property
    def profit(self) -> float:
        FUEL_COST = 0.4
        return self.money_made - (self.base_fuel * FUEL_COST)

    base_mass = 50_000
    base_fuel = 30_000

    @property
    def mass(self) -> float:
        asteroid_mass = sum([c.mass for c in self.carrying]) if self.carrying else 0
        return self.base_mass + self.fuel + asteroid_mass

    @property
    def distance_to_home(self) -> float:
        return self.position | self.base_station.position

    def fuel_to_get_to(self, target: Body) -> float:
        delta_v = self.base_station.delta_v_for(self)
        energy = 1 / 2 * self.mass * delta_v ** 2  # constant acceleration means this is all the energy we need
        # acceleration = G.value * self.mass * target.mass / (target.position | self.position ** 2)
        return energy / self.efficiency

    def travel_to(self, target: HasPosition) -> None:
        if isinstance(target, Asteroid):
            self.fuel -= self.fuel_to_get_to(target)
            remaining_energy = self.fuel * self.efficiency
            # rearrangement on E = m/2 * v^2 where m is a varible to be found and v is a function of m
            m = Symbol("m")
            (final_mass,) = solve(  # type: ignore
                ((2 * remaining_energy) / ((3300 * target.gravity) ** 2)) - (m * (log(m / 50_000)) ** 2),  # type: ignore
                m,
            )
            final_mass: float
            self.asteroid = target
            self.carrying = target.best_to_take_home(final_mass - self.base_mass)
        elif isinstance(target, Body):
            if self.carrying is not None:
                self.money_made += sum([c.price for c in self.carrying])
                self.carrying = None
            self.fuel -= self.fuel_to_get_to(target)

        position = target.position

        self.position = position  # type: ignore
