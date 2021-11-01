from __future__ import annotations

import csv
import itertools
import math
import random
import statistics
from operator import attrgetter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Generic,
    NamedTuple,
    ParamSpec,
    TypeVar,
    Union,
    get_type_hints,
    overload,
)

from astropy import units
from astropy.units import Quantity

from .station import Satellite, Sun, Body

if TYPE_CHECKING:
    from typing_extensions import Self

    from .models import Miner


ASTEROIDS: Final[Sequence[Asteroid]] = []

T = TypeVar("T")
P = ParamSpec("P")


class Material(Enum):
    class Info(NamedTuple):
        price: float  # price in USD per kg
        density: float  # kgm^-3

    GANG = Info(0, 2_500)  # Gangrenous materials have no economic value. Roughly the density of the Earth's crust
    H2O = Info(1_000, 1_000)  # SpaceX's Falcon 9 (used to deliver to the ISS) costs $2,720 per kilogram.
    SI = Info(8.1, 1_400)
    FE = Info(45.6, 7_300)
    PT = Info(33_000, 21_450)
    MG = Info(15, 1_740)
    O2 = Info(10, 1_400)
    NI = Info(18, 8_910)

    @property
    def price(self) -> float:
        return self.value.price

    @property
    def density(self) -> float:
        return self.value.density


class Contents(NamedTuple):
    material: Material
    volume: float

    @property
    def mass(self) -> float:
        return self.material.density * self.volume

    @property
    def price(self) -> float:
        return self.material.price * self.mass

    @property
    def price_to_density_ratio(self) -> float:
        return self.price / self.material.density


class Category(Enum):
    value: tuple[tuple[int], list[Material]]

    def likely_contents(self, volume: float):
        _, materials = self.value
        proportions = [
            1 / (i + 2) for i in range(len(materials))
        ]  # scaled falling off of the proportions using an offset y=1/x graph
        normaliser = 1 / sum(proportions)
        return [
            Contents(
                material,
                volume * proportion * normaliser,  # makes the volume total the correct value
            )
            for material, proportion in zip(materials, proportions)
        ]

    @property
    def pv(self) -> float:
        return self.value[0][0]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name} eta={self.pv}>"


class Type:
    """The Tholen type.

    - The first numbers are the mean geometric albedo's of 784 different asteroids. [1]
    - The second number is the infrared beaming parameter for the

    [1] Source: https://iopscience.iop.org/article/10.1088/0004-637X/762/1/56#apj452366app1
    """

    class C(Category):
        # https://en.wikipedia.org/wiki/C-type_asteroid
        B = (
            (0.113,),
            [Material.SI, Material.H2O, Material.GANG, Material.FE],
        )  # https://en.wikipedia.org/wiki/B-type_asteroid#Characteristics
        C = (0.061,), [Material.GANG, Material.H2O, Material.FE]
        F = (
            (0.058,),
            [Material.FE, Material.GANG],
        )  # "F-type asteroids have spectra generally similar to those of the B-type asteroids, but lack ... hydrated minerals"
        G = (0.073,), [Material.GANG]  # https://en.wikipedia.org/wiki/G-type_asteroid  # check Ceres is in this

    class S(Category):
        # https://en.wikipedia.org/wiki/S-type_asteroid
        A = (
            (0.282,),
            [Material.O2, Material.SI, Material.MG],
        )  # https://en.wikipedia.org/wiki/A-type_asteroid
        R = (0.277,), [Material.SI, Material.GANG]  # https://en.wikipedia.org/wiki/R-type_asteroid
        S = (0.213,), [Material.GANG, Material.GANG]  # haven't found info on these yet.
        O = (0.256,), [Material.GANG, Material.GANG]
        K = (
            (0.143,),
            [Material.SI, Material.GANG, Material.H2O, Material.GANG],
        )  # https://en.wikipedia.org/wiki/K-type_asteroid

    class X(Category):
        E = (0.559,), [Material.FE, Material.SI, Material.MG]
        M = (0.175,), [Material.FE, Material.NI, Material.GANG, Material.PT, Material.GANG]
        P = (
            (0.049,),
            [Material.SI, Material.GANG, Material.H2O, Material.GANG],
        )  # https://en.wikipedia.org/wiki/P-type_asteroid

    CATEGORIES = tuple(itertools.chain.from_iterable(Category.__subclasses__()))

    @classmethod
    def from_(cls, data: Asteroid.Data) -> Category:
        # TODO https://www.aanda.org/articles/aa/pdf/2018/04/aa31806-17.pdf
        # for a lot of data on this.
        dist = statistics.NormalDist(
            data.eta,  # mean infra red beaming param
            statistics.fmean(
                (data.eta1sigu, data.eta1sigl, data.eta3sigu, data.eta3sigl)
            ),  # mean of the upper and lower sigma bounds
        )
        return min(
            cls.CATEGORIES,
            key=lambda category: abs(  # the minimum absolute difference between the geometric albedos
                category.pv - data.pv
            ),
        )


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


@dataclass(slots=True, repr=False)
class Asteroid(Satellite, Body):
    orbit_distance: float
    orbit_period: timedelta

    class Data(NamedTuple):
        """Data currently loaded from the data.csv file"""

        desig: str  # MPC Designation
        name: Union[str, None]  # Target Name
        number: Union[int, None]  # MPC Number
        survey: str  # Spitzer Survey
        a: Quantity[units.au]  # Semi-Major Axis
        e: float  # Eccentricity
        i: float  # Not sure what this is
        node: Quantity[units.degree]  # Ascending Node
        argper: Quantity[units.degree]  # Argument of the Periapsis
        period: Quantity[units.year]  # Orbital Period
        ra: Quantity[units.degree]  # Target RA at Midtime (J2000)
        dec: Quantity[units.degree]  # Target Dec at Midtime (J2000)
        vmag: Quantity[units.mag]  # Predicted Target V Magnitude
        absmag: Quantity[units.mag]  # Absolute Magnitude in V
        absmagsig: Quantity[units.mag]  # 1 sigma Uncertainty
        slopepar: float  # Photometric Slope Parameter (H-G)
        heldist: Quantity[units.au]  # Heliocentric Distance at Midtime
        obsdist: Quantity[units.au]  # Distance from Spitzer at Midtime
        alpha: Quantity[units.degree]  # Solar Phase Angle at Midtime
        elong: Quantity[units.degree]  # Solar Elongation at Midtime
        glxlon: Quantity[units.degree]  # Galactic Longitude at Midtime
        glxlat: Quantity[units.degree]  # Galactic Latitude at Midtime
        ra3sig: Quantity[units.arcsec]  # 3 sigma Uncertainty in RA
        dec3sig: Quantity[units.arcsec]  # 3 sigma Uncertainty in Dec
        midtime: datetime  # Observation Midtime (UT)
        midtimejd: float  # Observation Midtime (UT)
        aorkey: int  # Observation AOR Key
        framet: Union[float, None]  # Frame Time
        totalt: Union[timedelta, None]  # Total Integration Time
        elapsed: Union[timedelta, None]  # Total Elapsed Time
        notes: Union[str, None]  # Data Reduction Notes
        ch1: Union[Quantity[units.uJy], None]  # IRAC CH1 Flux Density
        ch1err: Union[Quantity[units.uJy], None]  # CH1 Flux Density Uncertainty
        ch1snr: Union[float, None]  # CH1 Signal-to-Noise Ration
        ch2: Quantity[units.uJy]  # IRAC CH2 Flux Density
        ch2err: Quantity[units.uJy]  # CH2 Flux Density Uncertainty
        ch2snr: float  # CH2 Signal-to-Noise Ration
        diam: Quantity[units.km]  # Volume-equ. Spherical Diameter
        d1sigl: Quantity[units.km]  # Diameter 1 sigma Interval Bottom
        d1sigu: Quantity[units.km]  # Diameter 1 sigma Interval Top
        d3sigl: Quantity[units.km]  # Diameter 3 sigma Interval Bottom
        d3sigu: Quantity[units.km]  # Diameter 3 sigma Interval Top
        pv: float  # Geometric Albedo (V-Band)
        pv1sigl: float  # Albedo 1 sigma Interval Bottom
        pv1sigu: float  # Albedo 1 sigma Interval Top
        pv3sigl: float  # Albedo 3 sigma Interval Bottom
        pv3sigu: float  # Albedo 3 sigma Interval Top
        eta: float  # Infrared Beaming Parameter
        eta1sigl: float  # Eta 1 sigma Interval Bottom
        eta1sigu: float  # Eta 1 sigma Interval Top
        eta3sigl: float  # Eta 3 sigma Interval Bottom
        eta3sigu: float  # Eta 3 sigma Interval Top
        reflsolunits: float  # CH2 Reflected Solar Fraction

        __repr__ = object.__repr__  # intentionally neuter the repr for easier debugging

    data: Data
    type: Category
    bound_to = Sun
    miner: Miner = ...
    # cached slots
    _contents_cs = ...
    _radius_cs = ...
    _volume_cs = ...
    _price_cs = ...
    _mass_cs = ...
    _image_cs = ...

    def __repr__(self):
        attrs = ("identifier", "type", "contents", "position", "orbit_distance", "orbit_period")
        resolved = [f"{name}={getattr(self, name)!r}" for name in attrs]
        return f"<{self.__class__.__name__} {', '.join(resolved)}>"

    @cached_slot_property
    def contents(self) -> list[Contents]:
        return self.type.likely_contents(self.volume)

    @cached_slot_property  # type: ignore
    def radius(self) -> float:
        """The radius of the asteroid assuming it is a perfect sphere."""
        return self.data.diam.si.value / 2

    @cached_slot_property
    def volume(self) -> float:
        """The volume of the asteroid assuming it is a perfect sphere."""
        return 4 / 3 * math.pi * self.radius ** 3

    @property
    def identifier(self) -> str | int:
        """A unique identifier for each asteroid."""
        return self.data.name or self.data.number or self.data.desig

    @cached_slot_property
    def price(self) -> float:
        """The total price of the asteroid."""
        return sum([c.price for c in self.contents])

    @cached_slot_property  # type: ignore
    def mass(self) -> float:
        """The total mass of the asteroid."""
        # density / volume = mass
        return sum([c.mass for c in self.contents])

    def best_to_take_home(self, within_mass: float) -> list[Contents]:
        best: list[Contents] = []
        for content in sorted(self.contents, key=attrgetter("price_to_density_ratio")):
            if content.material == Material.GANG:
                return best

            within_mass -= content.mass
            if within_mass <= 0:
                if within_mass != 0:  #  only take some of the contents back
                    total_mass = content.material.density * content.volume
                    best.append(
                        Contents(content.material, volume=(total_mass + within_mass) * content.material.density)
                    )
                return best
            best.append(content)

        return best

    @cached_slot_property
    def image(self) -> str:
        """Retrieve an image of the asteroid."""
        return self._request()

    @classmethod
    def _request(cls, query: str) -> Any:
        ...

    @classmethod
    def random(cls) -> Asteroid:
        return random.choice(ASTEROIDS)
        info = cls._request(
            """
        SELECT * FROM table_name
        ORDER BY RAND()
        LIMIT 1;
        """
        )
        return info


def convert(type: type, value: str) -> Any:
    if getattr(type, "__origin__", None) is Quantity:
        return Quantity(value, type.__metadata__[0])  # type: ignore
    if type == Union[str, None]:
        return value or None
    if type is int:
        return int(value)
    if type == Union[int, None]:
        return int(value) if value else None
    if type is float:
        return float(value)
    if type == Union[float, None]:
        return float(value) if value else None
    if type is datetime:
        return datetime.fromisoformat(value)
    if type is timedelta:
        return timedelta(seconds=float(value))

    return value


def load():
    global ASTEROIDS
    ASTEROIDS += [  # type: ignore
        Asteroid(
            type=Type.from_(asteroid_data),
            orbit_period=timedelta(seconds=asteroid_data.period.si.value),
            orbit_distance=asteroid_data.heldist.si.value,
            data=asteroid_data,
        )
        for asteroid_data in (
            Asteroid.Data(
                *[
                    convert(annotation, value)
                    for value, annotation in zip(line, get_type_hints(Asteroid.Data, include_extras=True).values())
                ]
            )
            for line in csv.reader(open("miner/data.csv"))
        )
    ]
