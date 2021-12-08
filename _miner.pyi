from astropy.coordinates import SkyCoord as _SkyCoord
from astropy.time import Time as _Time

def determine_perihelion(
    original_r: float,
    eccentricity: float,
    period: float,
    major_axis: float,
    midtime: float,
) -> tuple[str, float]: ...
def get_spitzer(time: _Time) -> _SkyCoord: ...
