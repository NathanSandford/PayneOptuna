import numpy as np
import torch
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system
from astropy.coordinates import UnitSphericalRepresentation, CartesianRepresentation

def ensure_tensor(input_, precision=torch.float32):
    if isinstance(input_, torch.Tensor):
        return input_.to(precision)
    elif isinstance(input_, np.ndarray):
        return torch.from_numpy(input_).to(precision)
    elif isinstance(input_, (int, float)):
        return torch.Tensor([input_]).to(precision)
    elif isinstance(input_, list):
        return torch.Tensor(input_).to(precision)
    else:
        raise TypeError(f"input_ type ({type(input_)}) cannot be converted to a Tensor")


def j_nu(x, nu, n_tau=100):
    x_ = x.unsqueeze(-1)
    tau = torch.linspace(0, np.pi, n_tau).view(1,-1)
    integrand = torch.cos(nu*tau - x_ * torch.sin(tau))
    return (1/np.pi) * torch.trapz(integrand, tau[0], dim=-1)


def thin_plate_spline(r):
    return r**2*torch.log(r)


def log_lambda_grid(dv, min_wave, max_wave):
    max_log_dv = np.log10(dv / 2.99792458e5 + 1.0)
    log_min_wave = np.log10(min_wave)
    log_max_wave = np.log10(max_wave)
    min_n_pixels = (log_max_wave - log_min_wave) / max_log_dv
    n_pixels = 2
    while n_pixels < min_n_pixels:
        n_pixels *= 2
    log_dv = (log_max_wave - log_min_wave) / (n_pixels - 1)
    wave = 10 ** (log_min_wave + log_dv * np.arange(n_pixels))
    return wave


def find_runs(value, a):
    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
    isvalue = np.concatenate(([0], np.equal(a, value).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isvalue))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def get_geomotion_correction(radec, time, longitude, latitude, elevation, refframe='heliocentric'):
    '''
    Lifted from PypeIt
    '''
    loc = (longitude * u.deg, latitude * u.deg, elevation * u.m,)
    obstime = Time(time.value, format=time.format, scale='utc', location=loc)
    # Calculate ICRS position and velocity of Earth's geocenter
    ep, ev = solar_system.get_body_barycentric_posvel('earth', obstime)
    # Calculate GCRS position and velocity of observatory
    op, ov = obstime.location.get_gcrs_posvel(obstime)
    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov
    if refframe == "heliocentric":
        # ICRS position and velocity of the Sun
        sp, sv = solar_system.get_body_barycentric_posvel('sun', obstime)
        velocity += sv
    # Get unit ICRS vector in direction of SkyCoord
    sc_cartesian = radec.icrs.represent_as(UnitSphericalRepresentation).represent_as(CartesianRepresentation)
    vel = sc_cartesian.dot(velocity).to(u.km / u.s).value
    vel_corr = np.sqrt((1. + vel / 299792.458) / (1. - vel / 299792.458))
    return vel_corr


def airtovac(wavelength):
    """ Convert air-based wavelengths to vacuum (from PypeIt)
    Parameters
    ----------
    wave: Quantity array
        Wavelengths
    Returns
    -------
    wave: Quantity array
        Wavelength array corrected to vacuum wavelengths
    """
    # Convert to AA
    wave = wavelength * u.AA
    # Standard conversion format
    sigma_sq = (1.e4/wavelength)**2.  # wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.)  # only modify above 2000A
    # Convert
    wavelength = wavelength*factor
    # Units
    new_wave = wavelength*u.AA
    new_wave.to(wave.unit)
    return new_wave.value


def noise_up_spec(D0, sigma_D0, factor, nspec=1, min_flux=-1e10, max_flux=1e10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sigma = np.sqrt(factor**2 - 1) * sigma_D0
    D = np.random.normal(
        D0, sigma,
        size=(nspec, D0.shape[0], D0.shape[1])
    ).clip(min=min_flux, max=max_flux)
    sigma_D = np.sqrt(sigma_D0**2 + sigma**2)
    return D, sigma_D


def pad_array(array, pad_size, pad_value):
    return torch.nn.functional.pad(
        array,
        pad=(pad_size.to(int), pad_size.to(int), 0, 0),
        value=pad_value,
    )


def unpad_array(array, pad_size):
    return array[..., pad_size.to(int):-pad_size.to(int)]
