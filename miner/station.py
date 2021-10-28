from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, ClassVar, TypeVar

from astropy import units
from astropy.constants import G, M_earth, M_sun, R_earth, R_sun, au
from astropy.units import Unit

from .position import HasPosition, Position

if TYPE_CHECKING:
    from .models import Miner

__all__ = (
    "Earth",
    "Moon",
    "Mars",
)

T = TypeVar("T")


class Station:
    position: ClassVar[Position]
    gravity: ClassVar[float]

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__


class Body:
    mass: float
    radius: float
    position: Position

    @property
    def gravity(self) -> float:
        return G.value * self.mass / self.radius ** 2

    def delta_v_for(self, miner: Miner) -> float:
        """Based on https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation"""
        lsp = 3300
        # value is the mean of https://en.wikipedia.org/wiki/Liquid_rocket_propellant#Bipropellants's LOX column.
        # "specific impulse is exactly proportional to exhaust gas velocity" -
        # https://en.wikipedia.org/wiki/Specific_impulse
        return lsp * self.gravity * math.log(miner.mass / miner.base_mass)


class Planet(Station, Body):
    orbit_distance: float
    orbit_period: timedelta

    @property  # type: ignore
    def position(cls) -> Position:
        now = datetime.now(tz=timezone.utc)
        jan_1st = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        percentage_complete = (now - jan_1st).total_seconds() / cls.orbit_period.total_seconds()
        angle = (2 * math.pi) * percentage_complete  # angle in radians
        # if you think of this from a bird's eye view and the x and y components similarly to a unit circle diagram this
        # makes more sense
        x = math.sin(angle) * cls.orbit_distance
        z = math.cos(angle) * cls.orbit_distance
        return Position(x, 0, z)  # assume the orbit is flat


class Satellite:
    """A Satellite is any object that has an orbit around another body."""

    bound_to: HasPosition
    orbit_distance: float
    orbit_period: timedelta

    @property
    def position(self) -> Position:
        main_station_position: Position = self.bound_to.position  # type: ignore
        now = datetime.now(tz=timezone.utc)
        jan_1st = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        current_cycle = jan_1st
        while now > current_cycle + self.orbit_period:  # get to the current position in the orbital cycle
            current_cycle += self.orbit_period

        percentage_complete = (now - current_cycle).total_seconds() / self.orbit_period.total_seconds()
        angle = 2 * math.pi * percentage_complete
        x = math.sin(angle) * self.orbit_distance
        z = math.cos(angle) * self.orbit_distance
        return Position(x, 0, z) + main_station_position


def instanciate(cls: type[T]) -> T:
    return cls()


# neat trick to instantiate the Moon class so it's position can be worked out in Satellite.position
@instanciate
class Sun(Body):
    # we use a heliocentric model as everything we are currently interested in mining orbits around the sun.
    position = Position.origin()
    mass: float = M_sun.value  # type: ignore
    radius: float = R_sun.value  # type: ignore


@instanciate
class Earth(Planet):
    orbit_distance: float = au.value  # type: ignore # 1 AU in m
    orbit_period = timedelta(days=365.25)
    mass: float = M_earth.value  # type: ignore
    radius: float = R_earth.value  # type: ignore


@instanciate
class Moon(Satellite, Station, Body):
    bound_to = Earth
    orbit_distance = 384_400_000
    orbit_period = timedelta(days=28)
    gravity = 1.62


@instanciate
class Mars(Planet):
    orbit_distance = 248_550_000_000
    orbit_period = timedelta(days=687)
    gravity = 3.72
