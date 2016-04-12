import subprocess
import astropy.io.fits as ft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os.path as op
import astropy.wcs as wcs
import warnings
import json


# log
def log(func):

    def wrapper(*args, **kw):
        print('--------------------------------------------------------------------------')
        begin_time = time.clock()
        print('@Running the function %s() at %s' % (func.__name__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        func(*args, **kw)
        end_time = time.clock()
        print('@The function takes %f seconds to complete' % (end_time-begin_time))
        print('--------------------------------------------------------------------------')
        return

    return wrapper


class Galaxy(object):

    def __init__(self, fl, centroid, centroid_mode='world'):
        """initialize"""
        # intermediate files
        self.__file = {
            'sky': fl,
            'galaxy': 'galaxy.fits',
            'check': 'check.fits',
            'truncate': 'truncate.fits'
        }
        # array data
        self.__data = {
            'sky': np.array([]),
            'galaxy_1pr': np.array([]),
            'galaxy_1.5pr': np.array([]),
            'galaxy_2pr': np.array([]),
            'truncate': np.array([])
        }
        # galaxy's attributes
        self.__centroid = {
            'pix': [],
            'world': []
        }
        self.__petrosianRadius = None
        self.__surfaceBrightness = None
        self.__meanSurfaceBrightness = None
        self.__background = None
        self.__circleApertureFlux = None
        # structural parameters
        self.__structural_parameters = {
            'gini': None,
            'asymmetry': None,
            'moment': None,
            'concentration': None
        }
        # flags
        self.__flag = {
            'get_pr': False,
            'get_gd': False,
            'get_bg': False,
            'get_sp': False,
            'cal_a': False,
            'cal_g': False,
            'cal_m': False,
            'cal_c': False,
        }
        # SExtractor output string
        self.__SExtractorOutput = ''
        # open the file and convert world coordinate to pix coordinate(if necessary)
        with ft.open(self.__file['sky']) as hdu:
            self.__hdu = hdu[0]
            self.__header = self.__hdu.header
            self.__data['sky'] = np.array(self.__hdu.data)
        if centroid_mode is 'world':
            self.__centroid['world'] = list(centroid)
            self.__centroid['pix'] = list(wcs.WCS(self.__header).all_world2pix(np.array([centroid]), 1)[0])
        elif centroid_mode is 'pix':
            self.__centroid['pix'] = list(centroid)
            self.__centroid['world'] = list(wcs.WCS(self.__header).wcs_pix2world(np.array([centroid]), 1))
        return

    # properties
    @property
    def header(self):
        return repr(self.__header)

    @property
    def centroid_coordinate(self):
        return json.dumps({
            'pix': self.__centroid['pix'],
            'world': self.__centroid['world']
        }, indent=4)

    @property
    def petrosian_radius(self):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        return self.__petrosianRadius

    @property
    def background(self):
        if not self.__flag['get_bg']:
            self.__get_background__()
        return json.dumps(self.__background, indent=4)

    @property
    def gini_parameter(self):
        if self.__flag['cal_g']:
            return self.__structural_parameters['gini']
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        _F = np.sort(self.__data['galaxy_1.5pr'].flatten())
        n = len(_F)
        diff = np.array([2*l-n-1 for l in range(n)])
        self.__structural_parameters['gini'] = np.sum(diff*_F)/(_F.mean()*n*(n-1))
        self.__flag['cal_g'] = True
        return self.__structural_parameters['gini']

    @property
    def moment_parameter(self):
        if self.__flag['cal_m']:
            return self.__structural_parameters['moment']
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        ptr = self.__petrosianRadius
        _F = np.sort(self.__data['galaxy_1pr'].flatten())[::-1]
        arg = np.argsort(self.__data['galaxy_1pr'].flatten())[::-1]
        dist = [((arg[t] // (2*ptr+1))-ptr)**2+((arg[t] % (2*ptr+1))-ptr)**2 for t in range(len(_F))]
        _M = _F*np.array(dist)
        for i in range(len(_F) - 1):
            _F[i + 1] += _F[i]
        bound = float(np.argwhere(_F > 0.2 * np.sum(self.__data['galaxy_1pr']))[0])
        self.__structural_parameters['moment'] = _M[:bound].sum()/_M.sum()
        self.__flag['cal_m'] = True
        return self.__structural_parameters['moment']

    @property
    def asymmetry_parameter(self):
        if self.__flag['cal_a']:
            return self.__structural_parameters['asymmetry']
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        _I = np.copy(self.__data['galaxy_1.5pr'])
        _I180 = np.rot90(_I, 2)
        return np.sum(abs(_I - _I180)) / (2 * np.sum(abs(_I)))

    @property
    def concentration_parameter(self):
        if self.__flag['cal_c']:
            return self.__structural_parameters['concentration']
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        n = len(self.__meanSurfaceBrightness)
        self.__circleApertureFlux = np.array([(2*(r+1)-1)**2 for r in range(n)])*self.__meanSurfaceBrightness
        r20 = float(np.argwhere(self.__circleApertureFlux > 0.2*self.__data['galaxy_1.5pr'].sum())[0])
        r80 = float(np.argwhere(self.__circleApertureFlux > 0.8*self.__data['galaxy_1.5pr'].sum())[0])
        self.__structural_parameters['concentration'] = np.log10(r80/r20)
        self.__flag['cal_c'] = True
        return self.__structural_parameters['concentration']

    @property
    def structural_parameters(self):
        if not self.__flag['get_sp']:
            self.__structural_parameters['gini'] = self.gini_parameter
            self.__structural_parameters['moment'] = self.moment_parameter
            self.__structural_parameters['asymmetry'] = self.asymmetry_parameter
            self.__structural_parameters['concentration'] = self.concentration_parameter
            self.__flag['get_sp'] = True
        return json.dumps(self.__structural_parameters, indent=4)

    @property
    def half_light_radius_in_pixels(self):
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        n = len(self.__meanSurfaceBrightness)
        f = np.copy(self.__meanSurfaceBrightness)*[(2*r+1.0)**2 for r in range(n)]
        return float(np.argwhere(f > np.sum(self.__data['galaxy_1pr'])*0.5)[0])

    # core functions
    @log
    def __get_petrosian_radius__(self):
        if self.__flag['get_pr']:
            return
        _rightDistance = np.array(self.__data['sky'].shape)-np.array(self.__centroid['pix'])
        _leftDistance = np.array(self.__centroid['pix'])
        _boxSize = np.min([_rightDistance, _leftDistance, [250, 250]])
        self.__surfaceBrightness = np.zeros(_boxSize)
        for y in np.arange(self.__centroid['pix'][0]-_boxSize+1, self.__centroid['pix'][0]+_boxSize):
            for x in np.arange(self.__centroid['pix'][1]-_boxSize+1, self.__centroid['pix'][1]+_boxSize):
                dist = max(abs(y-self.__centroid['pix'][0]), abs(x-self.__centroid['pix'][1]))
                self.__surfaceBrightness[min(dist, _boxSize-1)] += self.__data['sky'][y][x]
        self.__meanSurfaceBrightness = np.copy(self.__surfaceBrightness)
        for r in np.arange(_boxSize):
            if r == 0:
                self.__meanSurfaceBrightness[1] += self.__meanSurfaceBrightness[0]
            else:
                if r < float(int(_boxSize - 1)):
                    self.__meanSurfaceBrightness[r + 1] += self.__meanSurfaceBrightness[r]
                self.__surfaceBrightness[r] /= 8*r
                self.__meanSurfaceBrightness[r] /= (2*(r+1)-1)**2
        eta = self.__surfaceBrightness/self.__meanSurfaceBrightness
        self.__petrosianRadius = float(np.argwhere(eta < 0.2)[0])
        self.__flag['get_pr'] = True
        return

    @log
    def __get_galaxy_data__(self):
        if self.__flag['get_gd']:
            return
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        ct = self.__centroid['pix']
        ptr = self.__petrosianRadius
        self.__data['galaxy_1pr'] = np.copy(self.__data['sky'][ct[0]-ptr:ct[0]+ptr+1, ct[1]-ptr:ct[1]+ptr+1])
        self.__data['galaxy_1.5pr'] = np.copy(self.__data['sky'][ct[0]-ptr*1.5:ct[0]+ptr*1.5+1, ct[1]-ptr*1.5:ct[1]+ptr*1.5+1])
        self.__data['galaxy_2pr'] = np.copy(self.__data['sky'][ct[0]-ptr*2:ct[0]+ptr*2+1, ct[1]-ptr*2:ct[1]+ptr*2+1])
        self.__flag['get_gd'] = True
        return

    @log
    def __get_background__(self):
        if self.__flag['get_bg']:
            return
        _SExtractorProcess = subprocess.Popen('sextractor %s' % self.__file['sky'],
                                              shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _SExtractorProcess.wait()
        self.__SExtractorOutput = _SExtractorProcess.stdout.readlines()
        _backgroundIndex = self.__SExtractorOutput.index(b'\x1b[1M> Scanning image\n')-1
        _backgroundInformation = self.__SExtractorOutput[_backgroundIndex].split()
        self.__background = {
            'mean': float(_backgroundInformation[2]),
            'rms': float(_backgroundInformation[4]),
            'threshold': float(_backgroundInformation[7])
        }
        self.__flag['get_bg'] = True
        return

    # visualizing methods
    def show_initial_image(self):
        subprocess.Popen('~/bin/ds9 -scale mode zscale -zoom 0.25 %s' % self.__file['sky'], shell=True, executable='/bin/zsh')
        return

    def show_eta_curve(self):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        sns.tsplot(self.__surfaceBrightness/self.__meanSurfaceBrightness)
        plt.show()
        return

    def show_galaxy_image(self, path='../tmp/', name='galaxy_1pr.fits'):
        if op.exists(self.__file['galaxy']):
            subprocess.call('rm '+self.__file['galaxy'], shell=True, executable='/bin/zsh')
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        ft.writeto(self.__file['galaxy'], self.__data['galaxy_1pr'])
        subprocess.call('mv %s %s' % (self.__file['galaxy'], path+name), shell=True, executable='/bin/zsh')
        subprocess.Popen('~/bin/ds9 -scale mode zscale -zoom 2 %s' % path+name, shell=True, executable='/bin/zsh')
        return

    def show_truncate_image(self, crd, radius, crd_mode='pix', path='../tmp/', name='truncate.fits'):
        coordinate = {
            'pix': [],
            'world': []
        }
        if crd_mode is 'world':
            coordinate['world'] = list(crd)
            coordinate['pix'] = list(wcs.WCS(self.__header).all_world2pix(np.array([crd]), 1)[0])
        elif crd_mode is 'pix':
            coordinate['pix'] = list(crd)
            coordinate['world'] = list(wcs.WCS(self.__header).wcs_pix2world(np.array([crd]), 1))
        self.__data['truncate'] = self.__data['sky'][coordinate['pix'][0]-radius+1: coordinate['pix'][0]+radius,
                                                     coordinate['pix'][1] - radius + 1: coordinate['pix'][1] + radius]
        if op.exists(self.__file['truncate']):
            subprocess.call('rm ' + self.__file['truncate'], shell=True, executable='/bin/zsh')
        ft.writeto(self.__file['truncate'], self.__data['truncate'])
        subprocess.call('mv %s %s' % (self.__file['truncate'], path + name), shell=True, executable='/bin/zsh')
        subprocess.Popen('~/bin/ds9 -scale mode zscale -zoom 2 %s' % path + name, shell=True, executable='/bin/zsh')
        return


@log
def test():
    warnings.filterwarnings('ignore')
    catalog = pd.read_csv('list.csv')
    fits_directory = '/Users/franky/Desktop/type1cut/'
    w = []
    for i in range(1):
        ctl = catalog.ix[i]
        name = ctl.NAME1+'_r.fits'
        ct = [ctl.RA1, ctl.DEC1]
        gl = Galaxy(fits_directory+name, ct)
        w.append(gl.moment_parameter)
    print(w)


@log
def load():
    data = pd.read_table('COSMOS-mor-H.txt', sep=' ', index_col=0)
    fits = '/Users/franky/Desktop/check/sky.fits'
    gls = data[(data.HalfLightRadiusInPixels > 25) &
               (data.HalfLightRadiusInPixels < 30)]
    gls.index = range(len(gls))
    for i in [0, 1, 2]:
        sample = gls.ix[i]
        gl = Galaxy(fits, [sample.Y_IMAGE, sample.X_IMAGE], centroid_mode='pix')
        gl.show_galaxy_image()


if __name__ == '__main__':
    # test()
    load()
