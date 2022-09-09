from distutils.core import setup
from distutils.command.install import INSTALL_SCHEMES
from os import sys
import os
import re

for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']


from imp import find_module
#from importlib import find_module
try:
    find_module('numpy')
except:
    sys.exit('### Error: python module numpy not found')
    
try:
    find_module('astropy')
except:
    try:
        find_module('pyfits')
    except:
        sys.exit('### Error: Neither astropy nor pyfits found.')

try:
    find_module('matplotlib')
except:
    sys.exit('### Error: python module matplotlib not found')


#try: find_module('MySQLdb')
#except: sys.exit('### Error: python module MySQLdb not found')


verstr = "unknown"
try:
    parentdir = os.getcwd()+'/'
    verstrline = open(parentdir+'cdfutils/_version.py', "rt").read()
except EnvironmentError:
    pass # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in " + parentdir + "cdfutils/_version.py")


setup(
    name = 'cdfutils',
    #version = verstr,#'0.1.3',
    author = 'Chris Fassnacht',
    author_email = 'cdfassnacht@ucdavis.edu',
    scripts=[],
    #url = 'lcogt.net',
    license = 'LICENSE',
    description = 'Base data handling functions used by several other packages',
    #long_description = open('README.txt').read(),
    requires = ['numpy','astropy','matplotlib'],
    packages = ['cdfutils'],
    #package_dir = {'':'src'}
    #package_data = {'SpecIm' : ['Data/*fits']}
)
