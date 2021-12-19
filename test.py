from datetime import datetime, timezone
from typing import Callable
from astropy.coordinates.funcs import get_sun
from astropy.coordinates.representation import CartesianDifferential
from astropy.coordinates.solar_system import get_body, get_body_barycentric_posvel
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation
from astropy import units
from poliastro.iod.vallado import lambert
from poliastro.maneuver import Maneuver
from poliastro.twobody.orbit import Orbit
from miner.asteroid import load, ASTEROIDS
from poliastro.plotting import OrbitPlotter3D
from poliastro.constants import J2000
from poliastro.bodies import Earth as PoliEarth, Sun
from plotly.graph_objs._figure import Figure
from datetime import timedelta
from miner import Miner, Earth
from miner.utils import get_earth


def plot_orbits(identifier):
    load()

    miner = Miner(Earth)
    asteroid = next(asteroid for asteroid in ASTEROIDS if str(asteroid.identifier) == identifier)

    def plot_pair(time, label):
        plotter.plot_body_orbit(PoliEarth, Time(time, scale="tdb"), label=f"Earth {label}")
        plotter.plot(
            asteroid.orbit.propagate((time - asteroid.data.midtime).total_seconds() * units.s),
            label=f"{identifier} {label}",
        )

    now = datetime.now(tz=timezone.utc)
    plotter = OrbitPlotter3D()

    plot_pair(now, "now")
    miner.travel_to(asteroid)

    plot_pair(miner.time_set_off, "at takeoff from Earth")
    plot_pair(miner.time_at_arrival, f"at arrival to {identifier}")
    plotter.plot_body_orbit
    # # r, v = get_body_barycentric_posvel("earth", Time(miner.time_set_off))
    # # coordinates =

    # # destination_frame = _get_destination_frame(attractor, plane, epochs)
    # from poliastro.ephem import Ephem

    # earth_orbit = Orbit.from_body_ephem(PoliEarth, epoch=Time(miner.time_set_off))
    # plotter.plot(earth_orbit, label="The start position")
    # position_at_arrival = asteroid.orbit.propagate(
    #     (miner.time_at_arrival - asteroid.data.midtime).total_seconds() * units.s
    # )
    # plotter.plot_maneuver(
    #     earth_orbit,
    #     Maneuver.lambert(earth_orbit, position_at_arrival, method=lambert),
    #     color="purple",
    # )

    miner.time_at_arrival += timedelta(days=120)

    miner.travel_to(miner.base_station)
    plot_pair(miner.time_set_off, f"at takeoff from {identifier}")
    plot_pair(miner.time_at_arrival, "at arrival back to Earth")

    fig: Figure = plotter.show()
    fig.show()


plot_orbits("363067")
