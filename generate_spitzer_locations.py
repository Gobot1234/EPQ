import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from astropy.coordinates import CartesianRepresentation, get_sun
from astropy.time import Time
from typing_extensions import Self

from miner.position import Position
from miner.utils import get_earth

CWD = Path(__file__).parent
# https://ssd.jpl.nasa.gov/horizons/app.html
FILE = CWD / "horizons_results.csv"
# Date__(UT)__HR:MN:SC.fff, R.A.___(ICRF), DEC____(ICRF),    hEcl-Lon,  hEcl-Lat,
# %Y-%b-%d %X

GENERATED = CWD / "miner" / "spitzer_location.csv"


if __name__ == "__main__":
    with FILE.open() as to_read, GENERATED.open("w+") as to_write:
        writer = csv.writer(to_write)
        x = []
        y = []
        z = []
        ax = plt.axes(projection="3d")
        for line in csv.reader(to_read):
            dt_fmt, _, __, ra, dec, geo_dist, ___, ____ = line
            dt = datetime.strptime(
                dt_fmt.rpartition(".")[0], " %Y-%b-%d %X"
            )  # lose millisecond precision here, oh well, it won't make much difference
            writer.writerow((dt.isoformat(), ra, dec, geo_dist))
            x.append(dt.timestamp())
            y.append(float(dec))
            z.append(float(ra))

        ax.plot3D(x, y, z)

        # sun: CartesianRepresentation = get_sun(time).cartesian  # type: ignore
        # earth: CartesianRepresentation = get_earth(time).cartesian  # type: ignore
        # earth_position = Position(earth.x.si.value, earth.y.si.value, earth.z.si.value)  # type: ignore
        # sun_position = Position(sun.x.si.value, sun.y.si.value, sun.z.si.value)  # type: ignore
        # print(earth_position, ra, dec, sun_position, sun_long, sun_lat)

        # sun_long_rad = math.radians(float(sun_long))
        # sun_lat_rad = math.radians(float(sun_lat))

        # math.cos(sun_long_rad)
        # math.sin(sun_long_rad)

        # # The coordinates of a vector that pass through X_0, Y_0, Z_0 and have directions a, b, c respectively is
        # # given by:
        # # X = X_0 + T x a
        # # Y = Y_0 + T x b
        # # Z = Z_0 + T x c

        # # At an intersection between the two lines the coordinates the vectors have equal x, y and z values.
        # # If they don't quite meet we make them meet when they are at their closest values (this is probably
        # # due to the angles not being high enough tolerance).

        # system = Matrix(
        #     # X_e + T_earth x Dir_earth_X = X_s + T_sun x Dir_sun_X = x
        #     [earth_position.x, T_earth * earth_direction_x, x],
        #     [sun_position.x, T_sun * sun_direction_x, x],
        #     # Y_e + T_earth x Dir_earth_Y = Y_s + T_sun x Dir_sun_Y = y
        #     [earth_position.y, T_earth * earth_direction_y, y],
        #     [sun_position.y, T_sun * sun_direction_y, y],
        #     # Z_e + T_earth x Dir_earth_Z = Z_s + T_sun x Dir_sun_Z = z
        #     [sun_position.z, T_sun * sun_direction_z, z],
        #     [earth_position.z, T_earth * earth_direction_z, z],
        # )

        ax.set_xlabel("Time (since the unix epoch) / s")
        ax.set_ylabel("Declination / ˚")
        ax.set_zlabel("Rising Ascension / ˚")

        plt.savefig("Spitzer Rising Ascension and Declination with respect to time.png")
        # plt.savefig("Spitzer Geocentric-distance with time.png")
