from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING, Any

from astropy.constants import G
from astropy import units
from scipy.optimize import minimize_scalar, root_scalar

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
    elapsed_time: timedelta = field(default_factory=timedelta)
    time_set_off: datetime = field(default_factory=partial(datetime.now, tz=timezone.utc))
    time_at_arrival: datetime = field(default_factory=partial(datetime.now, tz=timezone.utc))
    distance_travelled: float = 0.0
    efficiency: float = 45_000_000  # conversion rate of the 1kg of fuel to energy in Joules.
    # https://en.wikipedia.org/wiki/Energy_density#In_chemical_reactions_(oxidation)
    base_mass: float = 5_000.0
    fuel: float = Any  # the mass of fuel the miner has in kg.
    stage_one_fuel = 407_000
    stage_two_fuel = 107_200
    base_fuel: float = (
        stage_one_fuel + stage_two_fuel
    )  # TODO maybe optimize for wet amount using the rearrangement on wikipedia
    # https://www.spacelaunchreport.com/falconH.html
    current_stage_final_mass: float = Any

    def __post_init__(self) -> None:
        self.position = self.base_station.position
        self.fuel = self.base_fuel
        self.current_stage_final_mass = self.stage_two_fuel + self.base_mass

    @property
    def profit(self) -> float:
        FUEL_COST = 1.05  # https://www.globalpetrolprices.com/kerosene_prices (0.85/L and 1L=0.817)
        return self.money_made - (self.base_fuel * FUEL_COST)

    @property
    def mass(self) -> float:
        asteroid_mass = sum(c.mass for c in self.carrying) if self.carrying else 0
        return self.base_mass + self.fuel + asteroid_mass

    @property
    def distance_to_home(self) -> float:
        return self.position | self.base_station.position

    def fuel_to_get_to(self, target: HasPosition) -> float:
        delta_v_for_asteroid = getattr(self.asteroid, "delta_v_for", None)
        delta_v = (delta_v_for_asteroid or self.base_station.delta_v_for)(self)

        energy = 1 / 2 * self.mass * delta_v ** 2  # to escape station's gravity
        base_position = self.position
        print("Fuel for 1st stage is", energy / self.efficiency)

        now = self.time_at_arrival
        final_position: Position
        distance: float
        time: datetime
        delta: timedelta

        # TODO actually make the time taken to get there count?
        def distance_at(x: float) -> float:  # optimise to wait for the shortest distance to the asteroid
            nonlocal time, final_position, distance, delta
            delta = timedelta(seconds=x)
            time = now + delta
            final_position = target.position_at(time)
            distance = base_position | final_position
            return distance / delta_v

        res = minimize_scalar(
            distance_at,
            bounds=(
                0,
                20 * 365 * 24 * 60 * 60,
            ),  # some orbits have a very large synodic periods as the periods are similar to the Earth's so this needs to be big
            method="bounded",
            options={"maxiter": 10 ** 100},
        )  # iterations over the eculidian distance between the orbits over a year for the next 20 years.
        print((distance * units.m).to(units.au))
        self.distance_travelled += distance
        self.time_set_off = time
        self.time_at_arrival = time + timedelta(seconds=res.x)
        self.elapsed_time += delta
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
            energy * 2 + gravitational_energy  # amount of energy required is same for both take off and landing
        ) / self.efficiency

    def travel_to(self, target: HasPosition) -> None:
        if isinstance(target, Asteroid):
            fuel = self.fuel_to_get_to(target)
            print(f"Fuel required to get to {target.identifier} is", fuel)
            self.fuel -= fuel
            if self.fuel < 0:
                return print("Asteroid", target.identifier, "is too far away to travel to for the next 20 years")
            remaining_energy = self.fuel * self.efficiency

            def f(m: float) -> float:
                # rearrangement on E = m/2 * ∆v^2 where m is a varible to be found and v is a function of m (rocket eq.)
                # 0 = 2E / ∆v^2 - m
                # (-569 * atm + ve) * math.log(miner.mass / miner.current_stage_final_mass)
                return 2 * remaining_energy / (3410 * (math.log(self.mass / (m + self.base_mass)))) ** 2 - m

            solutions = root_scalar(
                f=f,
                # fprime=lambda x: float(f_prime.replace(m, x).evalf()),
                # fprime2=lambda x: float(f_prime_prime.replace(m, x).evalf()),
                # x0=10 ** 10,  # something very likely to be on the other side of the asymptote
                # options={"maxiter": sys.maxsize},
                bracket=(0.01, sys.maxsize),
            )  # there should only be one solution to this
            final_mass: float = solutions.root
            self.asteroid = target
            self.carrying = target.best_to_take_home(final_mass - self.mass)
        elif isinstance(target, Station):
            if self.carrying:
                self.money_made += sum(c.price for c in self.carrying)
                self.current_stage_final_mass = sum(c.mass for c in self.carrying) + self.base_mass
            try:
                fuel = self.fuel_to_get_to(target)
                self.fuel -= fuel
            except ValueError:
                fuel = None
                self.money_made = 0
            print("Fuel to get to the", self.base_station.__class__.__name__, "is", fuel)

        position = target.position

        self.position = position
