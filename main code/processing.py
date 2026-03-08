import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares


# this does the box least squares to find the best period
# time_array: array of time values
# flux_array: array of flux values
# pmin: minimum period
# pmax: maximum period
# qmin: minimum transit duration
# qmax: maximum transit duration
def bls_best(time_array, flux_array, pmin=0.5, pmax=100.0, qmin=0.0005, qmax=0.1):

    # uses box least squares mehods to detect an possible exoplanet (Astropy library)
    bls = BoxLeastSquares(time_array, flux_array)
    # gets possible periods
    periods = np.linspace(pmin, pmax, 5000).astype(np.float32)
    # gets possible durations
    durations = np.linspace(qmin, qmax, 20).astype(np.float32)
    # gets the powers
    power = bls.power(periods, durations)
    # finds the best period, duration, and transit time
    i = int(np.nanargmax(power.power))
    best_period = float(power.period[i])
    best_duration = float(power.duration[i])
    best_t0 = float(power.transit_time[i])
    return best_period, best_duration, best_t0


# folds the light curve to the best period and bins (divides) it into nbins segments
# time_array: array of time values
# flux_array: array of flux values
# period: period to fold the light curve
# t0: transit time
# nbins: number of bins to divide the light curve into
def fold_to_bins(time_array, flux_array, period, t0, nbins=512):
    # phase folds the light curve
    phase = ((time_array - t0 + 0.5 * period) % period) / period 
    # bins the light curve
    bins = np.linspace(0, 1, nbins + 1)
    # convert phase to bin indices 
    idx = np.digitize(phase, bins) - 1
    # compute median flux in each bin
    pf = np.array([np.nanmedian(flux_array[idx == k]) for k in range(nbins)], dtype=np.float32)
    # standardize
    pf = (pf - np.nanmedian(pf)) / (np.nanstd(pf) + 1e-6)
    return pf

