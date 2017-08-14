"""
A collection of fairly generic code for handling data
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

# ---------------------------------------------------------------------------


def sigclip(indata, nsig=3., mask=None, verbose=False):
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
        data = indata[mask]
    else:
        data = indata.copy()

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

    """ Error handling """
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


class DataGen(object):
    """

    The most generic data-handling methods.  Right now this is empty since
    the sigclip method, which had been in this class, has been moved out
    to be stand-alone since it was not working well as part of this class.

    """

    # -----------------------------------------------------------------------

    def __init__(self, data):
        self.data = np.asarray(data)

    # -----------------------------------------------------------------------

# ===========================================================================
#
# Start of Data1d class
#
# ===========================================================================


class Data1d(Table):
    """

    Code to perform actions on 1-dimesional data sets such as light curves or
    1d spectra.

    """

    # -----------------------------------------------------------------------

    def __init__(self, x, y, var=None, names=None):
        """

        Reads in the data, which can be expressed as y = y(x).

        """

        """ Set up the default names """
        if names is None:
            if var is None:
                names = ['x', 'y']
            else:
                names = ['x', 'y', 'var']

        """ Link to the inherited class """
        if var is None:
            Table.__init__(self, [x, y], names=names)
        else:
            Table.__init__(self, [x, y, var], names=names)

        """ Assign simple names to the columns """
        self.x = self[self.colnames[0]]
        self.y = self[self.colnames[1]]
        if var is not None:
            self.var = self[self.colnames[2]]
        else:
            self.var = None

    # -----------------------------------------------------------------------

    def fit_poly(self, fitorder, fitrange=None, nsig=3.0, y0=None, doplot=True,
                 markformat='bo', xlabel='x', ylabel='y', title=None):
        """

        This method is essentially just a pair of calls to numpy's polyfit
        and polyval.  However, there are a few add-ons, including the clipping
        of outliers (set by nsig), an optional limitation on the x range of
        the data to be fit (set by fitrange), and optional plotting of the
        resulting fit.

        """

        """ First a sigma clipping to reject clear outliers """
        if fitrange is None:
            tmpfitdat = self.y.data.copy()
        else:
            fitmask = np.logical_and(self.x >= fitrange[0],
                                     self.x < fitrange[1])
            tmpfitdat = self.y.data[fitmask]
        dmu, dsig = sigclip(tmpfitdat, nsig=nsig)
        goodmask = np.absolute(self.y.data - dmu) < nsig * dsig
        badmask = np.absolute(self.y.data - dmu) >= nsig * dsig
        xgood = self.x.data[goodmask]
        dgood = self.y.data[goodmask]
        xbad = self.x.data[badmask]
        dbad = self.y.data[badmask]

        """ Fit a polynomial to the trace """

        if fitrange is None:
            xpoly = xgood
            dpoly = dgood
        else:
            fitmask = np.logical_and(xgood >= fitrange[0], xgood < fitrange[1])
            xpoly = xgood[fitmask]
            dpoly = dgood[fitmask]

        if fitorder > -1:
            dpoly = np.polyfit(xpoly, dpoly, fitorder)
        elif y0 is not None:
            dpoly = np.array([y0, ])
        else:
            dpoly = None

        """
        Calculate the fitted function
        """
        fitx = np.arange(self.x.size)
        if dpoly is not None:
            fity = np.polyval(dpoly, fitx)
        else:
            fity is None

        """ Plot the results """
        ymin = dmu - 4.5*dsig
        ymax = dmu + 4.5*dsig
        if doplot and fity is not None:
            plt.plot(self.x, self.y.data, markformat)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            """ Show the value from the compressed spatial profile """
            if y0 is not None:
                plt.axhline(y0, color='k', linestyle='--')

            """ Mark the bad points that were not included in the fit """
            plt.plot(xbad, dbad, "rx", markersize=10, markeredgewidth=2)

            """ Show the fitted function """
            plt.plot(fitx, fity, "r")

            """
            Show the range of points included in the fit, if fitrange was set
            """
            if fitrange is not None:
                plt.axvline(fitrange[0], color='k', linestyle=':')
                plt.axvline(fitrange[1], color='k', linestyle=':')
                xtmp = 0.5 * (fitrange[1] + fitrange[0])
                xerr = xtmp - fitrange[0]
                ytmp = fity.min() - 0.2 * fity.min()
                plt.errorbar(xtmp, ytmp, xerr=xerr, ecolor="g", capsize=10)
            plt.xlim(0, self.x.max())
            plt.ylim(ymin, ymax)

        """
        Return the parameters produced by the fit and the fitted function
        """
        print dpoly
        return dpoly, fity
