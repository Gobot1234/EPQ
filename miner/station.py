from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, TypeVar

from astropy.constants import G, M_earth, M_sun, R_earth, R_sun, au
from astropy.coordinates import CartesianRepresentation, get_body, get_sun
from astropy.time import Time

from .position import HasPosition, Position

if TYPE_CHECKING:
    from .miner import Miner

__all__ = (
    "Earth",
    "Moon",
    "Mars",
)

T = TypeVar("T")


class Station:
    gravity: float

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__.lower()

    def position_at(self, dt: datetime) -> Position:
        raise NotImplementedError()

    @property
    def position(self) -> Position:
        return self.position_at(datetime.now(tz=timezone.utc))


class Body:
    mass: float
    radius: float
    position: Position

    @property
    def gravity(self) -> float:
        return G.value * self.mass / self.radius ** 2

    def delta_v_for(
        self,
        miner: Miner,
        *,
        atm: float,  # atmospheric pressure compared to the Earths.
        ve: float = 3510,
    ) -> float:
        """Based on https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation"""
        # value of ve is the mean of https://en.wikipedia.org/wiki/Liquid_rocket_propellant#Bipropellants's LOX column as
        # this is the fuel that the Falcon Heavy uses.
        # ∆y / ∆x = (3510 - 2941) / (1 - 0) = 569
        # assuming here that the change of ve is linear.
        # y = -569x + 2372
        return (-569 * atm + ve) * math.log(miner.mass / miner.current_stage_final_mass)


class Planet(Station, Body):
    orbit_distance: float
    orbit_period: timedelta
    atm: float

    def position_at(self, dt: datetime) -> Position:
        time = Time(dt)
        sky_coord = get_body(self.name, time)  # relative to the Earth, needs re-framing
        position: CartesianRepresentation = sky_coord.cartesian - get_sun(time).cartesian  # type: ignore
        return Position(position.x.si.value, position.y.si.value, position.z.si.value)  # type: ignore


class Satellite(Station):
    """A Satellite is any object that has an orbit around another body."""

    bound_to: HasPosition
    orbit_distance: float
    orbit_period: timedelta

    def position_at(self, dt: datetime) -> Position:
        main_station_position = self.bound_to.position_at(dt)
        position = Planet.position_at(self, dt)  # type: ignore
        return position + main_station_position


def instantiate(cls: Callable[[], T]) -> T:
    return cls()


# neat trick to instantiate the Moon class so it's position can be worked out in Satellite.position
@instantiate
class Sun(Body):
    # we use a heliocentric model as everything we are currently interested in mining orbits around the sun.
    position = Position.origin()
    mass: float = M_sun.value  # type: ignore
    radius: float = R_sun.value  # type: ignore

    def position_at(self, dt: datetime) -> Position:
        return self.position  # it's always at (0, 0, 0)


@instantiate
class Earth(Planet):
    orbit_distance: float = au.value  # type: ignore # 1 AU in m
    orbit_period = timedelta(days=365.25)
    mass: float = M_earth.value  # type: ignore
    radius: float = R_earth.value  # type: ignore
    atm: float = 1.0


@instantiate
class Moon(Satellite, Station, Body):
    bound_to = Earth
    orbit_distance = 384_400_000
    orbit_period = timedelta(days=28)
    gravity = 1.62
    atm: float = 3.0e-15  # basically nothing


@instantiate
class Mars(Planet):
    orbit_distance = 248_550_000_000
    orbit_period = timedelta(days=687)
    gravity = 3.72
    atm = 0.0060
