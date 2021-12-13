from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from functools import lru_cache, partial
from typing import Any, Final, Generic, ParamSpec, TypeVar, overload

from astropy import units
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation, get_body
from astropy.time import Time
from scipy import interpolate
from typing_extensions import Self

T = TypeVar("T")
P = ParamSpec("P")


class cached_slot_property(Generic[P, T]):
    """A decorator for properties that are very frequently lazily accessed."""

    def __init__(self, func: Callable[P, T]):
        self.__func__ = func

    @overload
    def __get__(self: Self, instance: None, _) -> Self:
        ...

    @overload
    def __get__(self, instance: Any, _) -> T:
        ...

    def __get__(self, instance: Any, _):  # type: ignore
        if instance is None:
            return self

        attr = f"_{self.__func__.__name__}_cs"
        result = getattr(instance, attr)
        if result is ...:
            result = self.__func__(instance)  # type: ignore
            setattr(instance, attr, result)
        return result


def get_earth(time: Time) -> UnitSphericalRepresentation:
    earth = get_body("earth", time)
    return UnitSphericalRepresentation.from_cartesian(earth.cartesian)  # (0, 0, 0)


LOCATIONS: Final[Mapping[datetime, SkyCoord]] = {}
INTERPOLATION_DISTANCE: partial[float] = Any
INTERPOLATION_RA: partial[float] = Any
INTERPOLATION_DEC: partial[float] = Any


@lru_cache(maxsize=64)  # this should mean that the results stay in the cache for subsequent calls.
def get_spitzer(time: Time) -> SkyCoord:
    """Get the location of Spitzer at a given time using spline interpolation within the mission time."""
    # assert EPOCH <= time <= END
    global INTERPOLATION_DISTANCE, INTERPOLATION_RA, INTERPOLATION_DEC

    if not LOCATIONS:
        with open("miner/spitzer_location.csv") as fp:
            for line in csv.reader(fp):
                dt_str, ra, dec, distance = line
                time_ = datetime.fromisoformat(dt_str)
                LOCATIONS[time_] = SkyCoord(  # type: ignore
                    ra=float(ra) * units.degree,
                    dec=float(dec) * units.degree,
                    distance=float(distance) * units.au,
                    obstime=time_,
                    equinox="J2000",
                )
        INTERPOLATION_DISTANCE = partial(  # type: ignore
            interpolate.splev,  # type: ignore
            tck=interpolate.splrep(  # type: ignore
                [time_.timestamp() for time_ in LOCATIONS],
                [sky_coord.distance.value for sky_coord in LOCATIONS.values()],
                s=0,
            ),
            der=0,
        )
        INTERPOLATION_RA = partial(  # type: ignore
            interpolate.splev,  # type: ignore
            tck=interpolate.splrep(  # type: ignore
                [time_.timestamp() for time_ in LOCATIONS],
                [sky_coord.ra.value for sky_coord in LOCATIONS.values()],
                s=0,
            ),
            der=0,
        )
        INTERPOLATION_DEC = partial(  # type: ignore
            interpolate.splev,  # type: ignore
            tck=interpolate.splrep(  # type: ignore
                [time_.timestamp() for time_ in LOCATIONS],
                [sky_coord.dec.value for sky_coord in LOCATIONS.values()],
                s=0,
            ),
            der=0,
        )

    timestamp: float = time.to_datetime(timezone.utc).timestamp()
    return SkyCoord(
        ra=INTERPOLATION_RA(timestamp) * units.degree,
        dec=INTERPOLATION_DEC(timestamp) * units.degree,
        distance=INTERPOLATION_DISTANCE(timestamp) * units.au,
    )
