from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from astropy.constants import G
from sympy import log
import sympy
from sympy.abc import m
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

from .asteroid import Asteroid, Contents
from .position import HasPosition
from .station import Planet, Station, Sun

if TYPE_CHECKING:
    from .position import HasPosition, Position


@dataclass(slots=True)
class Miner:
    base_station: Planet
    position: Position = Any
    carrying: list[Contents] | None = None
    asteroid: Asteroid | None = None

    money_made: float = 0.0
    efficiency: float = 43_000_000  # conversion rate of the 1kg of fuel to energy in Joules.
    # https://en.wikipedia.org/wiki/Energy_density#In_chemical_reactions_(oxidation)
    base_mass: float = Any
    fuel: float = Any  # the mass of fuel the miner has in kg.
    base_fuel: float = 407_000 + 107_200  # TODO maybe optimize for wet amount using the rearrangement on wikipedia
    current_stage_final_mass: float = Any
    # https://www.spacelaunchreport.com/falconH.html

    def __post_init__(self) -> None:
        self.position = self.base_station.position
        self.base_mass = 4_500
        self.fuel = self.base_fuel
        self.current_stage_final_mass = 407_000 + self.base_mass

    @property
    def profit(self) -> float:
        FUEL_COST = 0.4
        return self.money_made - (self.base_fuel * FUEL_COST)

    @property
    def mass(self) -> float:
        asteroid_mass = sum(c.mass for c in self.carrying) if self.carrying else 0
        return self.base_mass + self.fuel + asteroid_mass

    @property
    def distance_to_home(self) -> float:
        return self.position | self.base_station.position

    def fuel_to_get_to(self, target: HasPosition) -> float:
        assert hasattr(target, "orbit_period")
        delta_v_for_asteroid = getattr(self.asteroid, "delta_v_for", None)
        delta_v = (delta_v_for_asteroid or self.base_station.delta_v_for)(
            self, atm=0 if delta_v_for_asteroid is not None else self.base_station.atm
        )

        energy = 1 / 2 * self.mass * delta_v ** 2  # to escape station's gravity
        base_position = self.position
        self.base_mass = 4_500  # since we are performing a 2 stage escape we only have 107T of payload after leaving station's gravity

        now = datetime.now(tz=timezone.utc)
        final_position: Position
        distance: float
        time: datetime

        def func(x: float) -> float:  # optimise to wait for the shortest distance to the asteroid
            nonlocal time, final_position, distance
            time = now + timedelta(seconds=x)
            final_position = target.position_at(now + timedelta(seconds=x))
            distance = base_position | final_position
            return distance / delta_v

        minimize_scalar(
            func,
            bounds=(
                0,
                20 * 365 * 24 * 60 * 60,
            ),  # some orbits have a very large synodic periods as the periods are similar to the Earth's so this needs to be big
            method="bounded",
            options={"maxiter": 10 ** 100},
        )  # iterations over the eculidian distance between the orbits over a year for the next 20 years.

        self.position = final_position
        # here we assume change in mass is not significant
        #
        # asteroid
        # |---r1 \
        # |       \
        # |        \
        # |         \
        # |---r2 station.position
        # |      /
        # |---- /
        # sun
        #
        # from http://www.physbot.co.uk/gravity-fields-and-potentials.html
        # V = GM/r
        # therefore
        # ∆V = GM/r1 - GM/r2 (Jkg^-1)
        # then from gravitational potential
        # E = ∆Um

        gravitational_energy = abs(
            self.mass
            * G.value
            * Sun.mass
            * ((1 / (Sun.position | base_position)) - (1 / (Sun.position | final_position)))
        )

        return (
            (energy * 2) + gravitational_energy  # amount of energy required is same for both take off and landing
        ) / self.efficiency

    def travel_to(self, target: HasPosition) -> None:
        if isinstance(target, Asteroid):
            fuel = self.fuel_to_get_to(target)
            print(f"Fuel required to get to {target.identifier} is", fuel)
            self.fuel -= fuel
            if self.fuel < 0:
                return print("Asteroid", target.identifier, "is too far away to travel to for the next 20 years")
            remaining_energy = self.fuel * self.efficiency

            # rearrangement on E = m/2 * ∆v^2 where m is a varible to be found and v is a function of m (rocket eq.)
            # 0 = 2E / ∆v^2 - m
            # (-569 * atm + ve) * math.log(miner.mass / miner.current_stage_final_mass)
            f = 2 * remaining_energy / (3510 * (log(self.mass / (m + 4_500)))) ** 2 - m
            print(f)
            f_prime = f.diff()
            f_prime_prime = f_prime.diff()

            solutions = root_scalar(
                f=lambda x: float(f.replace(m, x).evalf()),
                # fprime=lambda x: float(f_prime.replace(m, x).evalf()),
                # fprime2=lambda x: float(f_prime_prime.replace(m, x).evalf()),
                # x0=10 ** 10,  # something very likely to be on the other side of the asymptote
                options={"maxiter": sys.maxsize},
                bracket=(-4500, sys.maxsize),
            )  # there should only be one solution to this
            print(solutions)
            final_mass: float = solutions.root
            self.asteroid = target
            self.carrying = target.best_to_take_home(final_mass - self.mass)
        elif isinstance(target, Station):
            if self.carrying:
                self.money_made += sum(c.price for c in self.carrying)
                self.current_stage_final_mass = sum(c.mass for c in self.carrying)
            try:
                self.fuel -= self.fuel_to_get_to(target)
            except ValueError:
                # fuel is too low to return to earth
                self.money_made = 0

        position = target.position

        self.position = position
