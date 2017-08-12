"""
A collection of fairly generic code for handling data
"""

import numpy as np


class DataGen(object):
    """

    The most generic data-handling methods

    """

    # -----------------------------------------------------------------------

    def __init__(self, data):
        self.data = data

    # -----------------------------------------------------------------------

    def sigma_clip(self, nsig=3., mask=None, verbose=False):
        """
        Runs a sigma-clipping on the data.  The code iterates over
        the following steps until it has converged:
         1. Compute mean and rms noise in the (clipped) data set
         2. Reject (clip) data points that are more than nsig * rms from
            the newly computed mean (using the newly computed rms)
         3. Repeat until no new points are rejected
         Once convergence has been reached, the final clipped mean and clipped
         rms are returned.

         Optional inputs:
          nsig    - Number of sigma from the mean beyond which points are
                    rejected.  Default=3.
          mask    - If some of the input data are known to be bad, they can
                     be flagged before the inputs are computed by including
                     a mask.  This mask must be set such that True
                     indicates good data and False indicates bad data
          verbose - If False (the default) no information is printed
        """

        """ Determine what the input data set is """
        if mask is not None:
            data = self.data[mask]
        else:
            data = self.data

        """ Report the number of valid data points """
        size = data[np.isfinite(data)].size
        if verbose:
            print " sigma_clip: Full size of data       = %d" % data.size
            print " sigma_clip: Number of finite values = %d" % size

        """
        Reject the non-finite data points and compute the initial values
        """
        d = data[np.isfinite(data)].flatten()
        mu = d.mean()
        sig = d.std()
        mu0 = d.mean()
        sig0 = d.mean()
        if verbose:
            print ''
            print 'npix = %11d. mean = %f. sigma = %f' % (size, mu, sig)

        """ Iterate until convergence """
        delta = 1
        clipError = False
        while delta:
            size = d.size
            if sig == 0.:
                clipError = True
                break
            d = d[abs(d - mu) < nsig * sig]
            mu = d.mean()
            sig = d.std()
            if verbose:
                print 'npix = %11d. mean = %f. sigma = %f' % (size, mu, sig)
            delta = size-d.size
        if clipError:
            print ''
            print 'ERROR: sigma clipping failed, perhaps with rms=0.'
            print 'Setting mu and sig to their original, unclipped, values'
            print ''
            mu = mu0
            sig = sig0

        """ Clean up and return the results and clean up """
        del data, d
        return mu, sig

# ===========================================================================
#
# Start of Data1d class
#
# ===========================================================================

class Data1d(DataGen):
    """

    Code to perform actions on 1-dimesional data sets such as light curves or
    1d spectra.

    """
