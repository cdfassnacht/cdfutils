"""
coords.py

A collection of functions modified from coords.c and fitswcs.c, or new
functions that act as wrappers for functions in MWA's wcs.py
"""

import numpy as np
from math import pi, sin, cos, atan2, sqrt
from astropy import wcs
from astropy import units as u
try:
    from astropy.io import fits as pf
except ImportError:
    import pyfits as pf
try:
    from astropy.coordinates import SkyCoord
except ImportError:
    from astropy.coordinates import ICRS as SkyCoord

"""
NB: imfuncs is only imported for one call to open_fits.
Consider making a new fitsfuncs file in the CDFutils repo to hold open_fits
 (and perhaps other future methods)
"""
try:
    from SpecIm import imfuncs as imf
except ImportError:
    import imfuncs as imf

# -----------------------------------------------------------------------


def radec_to_skycoord(ra, dec):
    """
    Converts a (RA, dec) pair into the astropy.coordinates SkyCoord
    format

    Required inputs:
     ra  - RA in one of three formats:
            Decimal degrees: ddd.ddddddd  (as many significant figures
              as desired)
            Sexigesimal:     hh mm ss.sss (as many significant figures
              as desired)
            Sexigesimal:     hh:mm:ss.sss (as many significant figures
              as desired)
     dec - Dec in one of three formats:
            Decimal degrees: sddd.ddddddd, where "s" is + or -
            Sexigesimal:     sdd mm ss.sss (as many significant figures
               as desired)
            Sexigesimal:     sdd:mm:ss.sss (as many significant figures
               as desired)
    """

    """ Get RA format """
    if type(ra) == float or type(ra) == np.float32 or \
            type(ra) == np.float64:
        rafmt = u.deg
    else:
        rafmt = u.hourangle

    """ Dec format is always degrees, even if in Sexigesimal format """
    decfmt = u.deg

    """ Do the conversion """
    radec = SkyCoord(ra, dec, unit=(rafmt, decfmt))
    return radec

# -----------------------------------------------------------------------


def make_header(radec, pixscale, nx, ny=None, rot=None):
    """

    Makes a header with wcs information.

    Inputs:
      radec    - The desired (RA, Dec) pair to be put into the CRVAL
                  keywords.
                  NOTE: This should be in the SkyCoord format defined in
                   astropy.coordinates.  To convert a "normal" pair of
                   numbers / sexigesimal strings to SkyCoord format, use
                   the radec_to_skycoord method from coords.py in CDFutils
      pixscale - Desired pixel scale in arcsec/pix
      nx       - image size along the x-axis --or-- if the image is square
                   (indicated by ny=None) then this is also the y-axis size
      ny       - [OPTIONAL] y-axis size, if different from the x-axis size
                   ny=None means that the two axes have the same size
      rot      - [OPTIONAL] desired rotation angle, in degrees E of N.
                   NOT IMPLEMENTED YET
    """

    """ Create a blank 2d WCS container """
    w = wcs.WCS(naxis=2)

    """ Get the image size and central pixel """
    cp1 = nx / 2.
    if ny is None:
        cp2 = cp1
    else:
        cp2 = ny / 2.

    """ Fill it in with appropriate values and save it """
    px = pixscale / 3600.
    w.wcs.crpix = [cp1, cp2]
    w.wcs.crval = [radec.ra.degree, radec.dec.degree]
    w.wcs.cdelt = [(-1.*px), px]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.equinox = 2000.
    # self.subim_wcs = w

    """ Convert to a fits header format """
    hdr = w.to_header()
    return hdr, w

# -----------------------------------------------------------------------


def rscale_to_cdmatrix(pixscale, rot, pixscale2=0.0, verbose=True):
    """
    Converts a pixel scale and rotation (in degrees) into a cd matrix,
    which is returned.

    Inputs:
      pixscale  - pixel scale in arcsec/pix (corresponds to CDELT1)
      rot       - rotation angle, IN DEGREES (North through East)
      pixscale2 - [optional] second pixel scale, in arcsec/pix, if
                    CDELT2 != CDELT1.  Default=0.0 means just use pixscale
      rot2      - [optional] second rotation if axes have different rotations.
                    Default=0.0 means that only one rotation is used.
    """

    """ Initialize, and convert rotations to arcsec """
    cdmatx = np.zeros((2, 2))
    if pixscale2 == 0.0:
        pixscale2 = pixscale
    if verbose:
        print('')
        print('Creating a new CD matrix with:')
        print('  pixel scales (arcsec/pix): %7.4f  %7.4f'
              % (pixscale, pixscale2))
        print('  rotation (degrees N->E):    %+7.2f' % (rot))

    rot *= pi / 180.

    """ Set the values """
    cdmatx[0, 0] = -1.0 * pixscale * cos(rot)
    cdmatx[1, 0] = -1.0 * pixscale * sin(rot)
    cdmatx[0, 1] = -1.0 * pixscale2 * sin(rot)
    cdmatx[1, 1] = pixscale2 * cos(rot)
    """ Convert cd matrix to degrees and return"""

    cdmatx /= 3600.
    return cdmatx

# -----------------------------------------------------------------------


def cdmatrix_to_rscale(cdmatrix, verbose=True):
    """
    Converts a CD matrix from a fits header to pixel scales and rotations.

    Inputs:
      cdmatrix - the CD matrix, as a numpy array:
                  [[CD1_1, CD1_2], [CD2_1, CD2_2]]
      verbose  - set to True (the default) for verbose output
    """

    """ Set up """
    from numpy import linalg

    """
    Find the determinant, which sets the sign used in the calculation
    Note that the CD matrix typically encodes a left-handed coordinate system,
    since RA increases going to the left.
    """

    if linalg.det(cdmatrix) < 0:
        cdsgn = -1
    else:
        cdsgn = 1
        if verbose:
            print "  WARNING: cdmatrix_to rscale"
            print "    Astrometry is for a right-handed coordinate system."
    if(verbose):
        print cdsgn

    """ Calculate the rotation """
    crota2 = atan2(-1.0 * cdmatrix[1, 0], cdmatrix[1, 1])

    """
    Now convert CDn_m headers into pixel scales, expresess as CDELTn values.
    Use the following:
      CDELT1 = sqrt(CD1_1^2 + CD2_1^2)
      CDELT2 = sqrt(CD1_2^2 + CD2_2^2).
    """

    cdelt1 = -1.0*sqrt(cdmatrix[0, 0]**2 + cdmatrix[0, 1]**2)
    cdelt2 = sqrt(cdmatrix[1, 0]**2 + cdmatrix[1, 1]**2)

    """ Return the calculated values """
    return cdelt1, cdelt2, crota2

# -----------------------------------------------------------------------


def update_cdmatrix(fitsfile, pixscale, rot, pixscale2=0.0,
                    hext=0, verbose=True):
    """
    A wrapper for rscale_to_cdmatrix.  In this case, however, the inputs
    include a fits filename, and the cdmatrix generated by the call to
    rscale_to_cdmatrix is written into the header cards of the input fits file.

    Inputs:
      fitsfile  - name of fits file for which the CD matrix will be updated
      pixscale  - pixel scale in arcsec/pix (corresponds to CDELT1)
      rot       - rotation angle, in DEGREES (North through East)
      pixscale2 - [optional] second pixel scale, in arcsec/pix, if
                    CDELT2 != CDELT1.  Default=0.0 means just use pixscale
      rot2      - [optional] second rotation if axes have different rotations.
                    Default=0.0 means that only one rotation is used.
      hext      - [optional] HDU extension if not the default of HDU 0
      verbose   - [optional] flag set to True (the default) for verbose output
    """

    """ Open the input file """
    hdulist = imf.open_fits(fitsfile, 'update')
    hdr = hdulist[hext].header

    """ Generate a CD matrix based on the inputs """
    cdmatx = rscale_to_cdmatrix(pixscale, rot, pixscale2, verbose)

    """ Transfer new CD matrix to fits header cards and save """
    hdr.update('cd1_1', cdmatx[0, 0])
    hdr.update('cd1_2', cdmatx[0, 1])
    hdr.update('cd2_1', cdmatx[1, 0])
    hdr.update('cd2_2', cdmatx[1, 1])
    hdulist.flush()
    del hdr, hdulist

# -----------------------------------------------------------------------


def sky_to_darcsec(alpha0, delta0, alpha, delta):
    """
    Given a central position on the sky (alpha0,delta0), where these
    coordinates are (for now) expected to be in decimal degrees, and
    one or more other positions (alpha,delta), returns the offsets betwee
    the central position and each of the other positions in arcsec
    on a projected plane.
    Formulas taken from AIPS Memo 27
    For now, this uses only the SIN projection.

    Inputs:
      alpha0 - RA of central position in decimal degrees
      delta0 - Dec of central position in decimal degrees
      alpha  - array containing other RAs in decimal degrees
      delta  - array containing other Decs in decimal degrees
    """

    """ Make sure that inputs are in numpy array format """
    a0 = np.atleast_1d(alpha0).copy()
    d0 = np.atleast_1d(delta0).copy()
    a = np.atleast_1d(alpha).copy()
    d = np.atleast_1d(delta).copy()

    """ Convert inputs into radians """
    degs2rad = np.pi / 180.
    for i in [a0, d0, a, d]:
        i *= degs2rad

    """
    Calculate the direction cosines depending on projection method.
    For now, sin is the only projection.
    """
    proj = 'sin'

    if proj.lower() == 'sin':
        L = np.cos(d) * np.sin(a - a0)
        M = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0)

    """ Convert direction cosines to arcseconds """
    x_offset = L * 3600. / degs2rad
    y_offset = M * 3600. / degs2rad

    return x_offset, y_offset

# -----------------------------------------------------------------------


def darcsec_to_dpix(dalpha, ddelta, cdmatrix):
    """
    Given a series of (RA, Dec) offsets in arcsec, represented by dalpha and
    ddelta, converts them to pixel offsets for a given fits image.  The
    conversion from arcsec to pixels is done via the CD matrix that has
    been obtained from the fits header.  In particular, if invcd is the
    multiplicative inverse of the cdmatrix, then

      dx = invcd[0,0]*dalpha + invcd[0,1]*ddelta
      dy = invcd[1,0]*dalpha * invcd[1,1]*ddelta

    Inputs:
      dalpha   - vector containing RA offsets in arcsec
      ddelta   - vector containing Dec offsets in arcsec
      cdmatrix - CD matrix obtained from the fits header, where:
                    cdmatx[0,0] = CD1_1
                    cdmatx[0,1] = CD1_2
                    cdmatx[1,0] = CD2_1
                    cdmatx[1,1] = CD2_2
    """

    """ Initialize """
    da = np.atleast_1d(dalpha)
    dd = np.atleast_1d(ddelta)
    if da.size != dd.size:
        print('')
        print('ERROR: darcsec_to_dpix. Input offset vectors must be the same'
              'size')
        return np.nan, np.nan
    dx = np.zeros(da.size)
    dy = np.zeros(dd.size)

    """ Convert CD matrix to arcsec/pix """
    cd = cdmatrix.copy() * 3600.

    """ Invert the CD matrix """
    from numpy import linalg
    invcd = linalg.inv(cd)

    """ Loop through the arcsec offsets and convert to pixel offsets """
    for i in range(da.size):
        dsky = np.array([da[i], dd[i]])
        dpix = np.dot(invcd, dsky)
        dx[i] = dpix[0]
        dy[i] = dpix[1]

    return dx, dy
