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

    def __init__(self, file, centroid, centroid_mode='world'):
        """initialize"""
        self.__skyFile = file
        # open the file and convert world coordinate to pix coordinate(if necessary)
        with ft.open(self.__skyFile) as hdu:
            self.__hdu = hdu[0]
            self.__header = self.__hdu.header
            self.__skyData = np.array(self.__hdu.data)
        if centroid_mode is 'world':
            self.__worldCentroid = list(centroid)
            self.__pixCentroid = list(wcs.WCS(self.__header).all_world2pix(np.array([centroid]), 1)[0])
        elif centroid_mode is 'pix':
            self.__pixCentroid = list(centroid)
            self.__worldCentroid = list(wcs.WCS(self.__header).wcs_pix2world(np.array([centroid]), 1))
        # galaxy's attributes
        self.__petrosianRadius = None
        self.__surfaceBrightness = None
        self.__meanSurfaceBrightness = None
        self.__galaxyData = None
        self.__background = None
        # intermediate files
        self.__truncateFile = 'truncate.fits'
        # flags
        self.__flag = {
            'get_pr': False,
            'get_gd': False,
            'get_bg': False,
            'cal_a': False,
            'cal_g': False,
            'cal_m': False,
            'cal_c': False,
        }
        # others
        self.__SExtractorOutput = ''

    # properties
    @property
    def header(self):
        return repr(self.__header)

    @property
    def centroid_coordinate(self):
        return json.dumps({
            'pix': self.__pixCentroid,
            'world': self.__worldCentroid
        }, indent=4)

    @property
    def petrosian_radius(self):
        return self.__petrosianRadius

    @property
    def background(self):
        if not self.__flag['get_bg']:
            self.__get_background__()
        return json.dumps(self.__background, indent=4)

    @property
    def asymmetry_parameter(self):
        if not self.__flag['get_pr']:
            self.__truncate__()
        ct = self.__pixCentroid
        ptr = self.__petrosianRadius
        _I = np.copy(self.__skyData[ct[0]-ptr*1.5:ct[0]+ptr*1.5+1, ct[1]-ptr*1.5:ct[1]+ptr*1.5+1])
        _I180 = np.rot90(_I, 2)
        return np.sum(abs(_I-_I180))/(2*np.sum(abs(_I)))

    # core functions
    @log
    def __get_petrosian_radius__(self):
        if self.__flag['get_pr']:
            return
        _rightDistance = np.array(self.__skyData.shape)-np.array(self.__pixCentroid)
        _leftDistance = np.array(self.__pixCentroid)
        _boxSize = np.min([_rightDistance, _leftDistance, [250, 250]])
        self.__surfaceBrightness = np.zeros(_boxSize)
        for y in np.arange(self.__pixCentroid[0]-_boxSize+1, self.__pixCentroid[0]+_boxSize):
            for x in np.arange(self.__pixCentroid[1] - _boxSize+1, self.__pixCentroid[1] + _boxSize):
                dist = max(abs(y-self.__pixCentroid[0]), abs(x-self.__pixCentroid[1]))
                self.__surfaceBrightness[dist] += self.__skyData[y][x]
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
        print('Petrosian Radius = %f' % self.__petrosianRadius)
        return

    @log
    def __truncate__(self):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        ct = self.__pixCentroid
        ptr = self.__petrosianRadius
        self.__galaxyData = np.copy(self.__skyData[ct[0]-ptr:ct[0]+ptr+1, ct[1]-ptr:ct[1]+ptr+1])
        self.__flag['get_gd'] = True
        return

    @log
    def __get_background__(self):
        _SExtractorProcess = subprocess.Popen('sextractor %s' % self.__skyFile,
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
        subprocess.Popen('ds9 -scale mode zscale -zoom 0.25 %s' % self.__skyFile, shell=True, executable='/usr/bin/zsh')
        return

    def show_eta_curve(self):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        sns.tsplot(self.__surfaceBrightness/self.__meanSurfaceBrightness)
        plt.show()
        return

    def show_truncate_image(self, path, name):
        if op.exists(self.__truncateFile):
            subprocess.call('rm '+self.__truncateFile, shell=True, executable='/usr/bin/zsh')
        if not self.__flag['get_gd']:
            self.__truncate__()
        ft.writeto(self.__truncateFile, self.__galaxyData)
        subprocess.call('mv %s %s' % (self.__truncateFile, path+name), shell=True, executable='/usr/bin/zsh')
        subprocess.Popen('ds9 -scale mode zscale -zoom 2 %s' % path+name, shell=True, executable='/usr/bin/zsh')
        return


def test():
    warnings.filterwarnings('ignore')
    catalog = pd.read_csv('list.csv')
    fits_directory = '/home/franky/Desktop/type2cut/'
    tmp_path = '/home/franky/Desktop/tmp/'
    # for i in range(len(catalog)):
    for i in range(1):
        ctl = catalog.ix[i]
        name = ctl.NAME2+'_r.fits'
        ct = [ctl.RA2, ctl.DEC2]
        gl = Galaxy(fits_directory+name, ct)
        gl.show_truncate_image(tmp_path, name)
        # gl.__get_background__()
    # with open('list.csv') as file:
    #     for line in file:
    #         print(line)


if __name__ == '__main__':
    test()
