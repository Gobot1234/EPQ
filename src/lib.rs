use std::f64::consts::PI;

use chrono::{DateTime, Duration, FixedOffset, TimeZone, Utc};
use lazy_static::lazy_static;
use pyo3::{prelude::*, types::PyDict, wrap_pyfunction};

// https://www.spitzer.caltech.edu/mission/fast-facts
const TOTAL_MASS: u16 = 950; // The total initial mass of the telescope.
const PROPELLENT_MASS: f32 = 15.6; // The mass of fuel for the telescope.
const EXTRA_MASS: f32 = 50.4; // The mass of the helium in the telescope.
const SPECIFIC_IMPULSE: u8 = 80; // https://www.diva-portal.org/smash/get/diva2:75729/FULLTEXT01.pdf value from page 2.
const G: f64 = 6.6743e-11; // The value of G obtained by CODATA in 2018.
const SUN_MASS: f64 = 1.988409870698051e+30; // The mass of the Sun obtained by IAU in 2015.

lazy_static! {
    static ref EPOCH: DateTime<FixedOffset> =
        DateTime::parse_from_rfc2822("Sun, 24 Aug 2003 06:39:00 0000").unwrap();  // Time Spitzer was in orbit.
    static ref CRYO_END: DateTime<FixedOffset> =
        DateTime::parse_from_rfc2822("Fri, 15 May 2009 00:00:00 0000").unwrap();
    static ref END: DateTime<FixedOffset> =
        DateTime::parse_from_rfc2822("Thu, 30 Jan 2020 10:37:00 0000").unwrap();
    // static ref MASS_FLOW_RATE: f32 = PROPELLENT_MASS / (END.signed_duration_since(EPOCH)).num_seconds();
}

#[inline]
fn duration_from_seconds(duration: f64) -> Duration {
    Duration::nanoseconds((duration * 10E9_f64).floor() as i64)
}

#[pyfunction]
fn determine_perihelion(
    original_r: f64,
    eccentricity: f64,
    period: f64,
    major_axis: f64,
    midtime: f64,
) -> (String, f64) {
    // r = a(1 - eccentricity * cos(E))
    let e = ((major_axis - original_r) / (major_axis * eccentricity)).acos();
    let m = e - (eccentricity * e.sin());
    let new_period = duration_from_seconds(period);

    let mut perihelion: DateTime<Utc> = Utc.timestamp_millis((midtime * 1000.).floor() as i64)
        - duration_from_seconds(period * m / (2. * PI));

    let now = Utc::now();

    let mut next_perihelion = perihelion + new_period;

    while next_perihelion < now {
        perihelion = next_perihelion;
        next_perihelion = perihelion + new_period;
    }
    return (perihelion.to_rfc3339(), m);
}

#[pyfunction]
fn get_spitzer(py: Python, time: PyObject) -> PyResult<&PyAny> {
    let astropy = py.import("astropy")?;
    let coordinates = astropy.getattr("coordinates")?;
    let earth = coordinates.getattr("get_body")?.call1(("earth", time))?;

    // "The telescope drifts away from us at about 1/10th of one astronomical unit per year."
    // - https://www.spitzer.caltech.edu/mission/clever-choice-of-orbit
    // "Spitzer orbits the Sun on almost the same path as Earth ... Spitzer moves slower than
    //  Earth, so the spacecraft drifts farther away from our planet each year"
    // - https://solarsystem.nasa.gov/news/513/10-things-spitzer-space-telescope

    // https://cdn.intechopen.com/pdfs/37528/InTech-Cold_gas_propulsion_system_an_ideal_choice_for_remote_sensing_small_satellites.pdf
    // Turns out none of this was necessary.
    // Get location from file
    // Then find the time at whenever using indexing
    // Interpolate using scipy or use physics?
    let SkyCoord = coordinates.getattr("SkyCoord")?;
    let dict = PyDict::new(py);

    let (ra, dec, distance, time) = (1, 2, 3, 4);

    dict.set_item("ra", ra)?;
    dict.set_item("dec", dec)?;
    dict.set_item("distance", distance)?;
    dict.set_item("obstime", time)?;
    dict.set_item("equinox", "J2000")?;

    return Ok(SkyCoord.call((), Some(dict))?);
}

#[pymodule]
fn _miner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(determine_perihelion, m)?)?;
    m.add_function(wrap_pyfunction!(get_spitzer, m)?)?;
    Ok(())
}
