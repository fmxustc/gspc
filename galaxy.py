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
import random


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

    # @log
    def __init__(self, sky, centroid, centroid_mode='world',
                 shell='/usr/bin/zsh', ds9='ds9', sextractor='sextractor', **field):
        """initialize"""
        # intermediate files
        self.__file = {
            'sky': sky,
            'galaxy': 'galaxy.fits',
            'catalog': 'catalog.txt',
            'detect': 'detect.fits',
            'truncate': 'truncate.fits',
            'segmentation': 'segmentation.fits',
            'background': 'background.fits',
            'tmp': 'tmp.fits'
        }
        # array data
        self.__data = {
            'sky': np.array([]),
            'galaxy': {
                '1pr': np.array([]),
                '1.5pr': np.array([]),
                '2pr': np.array([]),
                '2.5pr': np.array([])
            },
            'truncate': np.array([]),
            'segmentation': {},
            'background': {}
        }
        # galaxy's attributes
        self.__galaxy = None
        self.__centroid = {
            'pix': [],
            'world': []
        }
        self.__petrosianRadius = 0
        self.__surfaceBrightness = np.array([])
        self.__meanSurfaceBrightness = np.array([])
        self.__circleApertureFlux = np.array([])
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
            'get_gd': {},
            'get_bg': {},
            'get_sp': False,
            'get_seg': {},
            'cal_a': False,
            'cal_g': False,
            'cal_m': False,
            'cal_c': False,
        }
        # SExtractor output string
        self.__SExtractorOutput = ''
        # system software settings
        self.__sys = {
            'shell': shell,
            'ds9': ds9,
            'sex': sextractor
        }
        # field setting
        self.__field = field['field']
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
            self.__centroid['world'] = list(wcs.WCS(self.__header).wcs_pix2world(np.array([centroid]), 1)[0])
        self.__adjust_centroid_coordinate__()
        return

    # properties
    @property
    def header(self):
        return repr(self.__header)

    @property
    def centroid_coordinate(self):
        print(self.__centroid)
        return json.dumps({
            'pix': self.__centroid['pix'],
            'world': self.__centroid['world']
        }, indent=4, sort_keys=True)

    @property
    def petrosian_radius(self):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        return self.__petrosianRadius

    @property
    def background(self):
        self.__set_field__('background')
        detect_field = self.__field['background']['detect']
        truncate_field = self.__field['background']['truncate']
        if truncate_field not in self.__flag['get_bg']:
            self.__get_background_map__(detect_field=detect_field, truncate_field=truncate_field)
        return json.dumps(self.__background, indent=4, sort_keys=True)

    @property
    def gini_parameter(self):
        if self.__flag['cal_g']:
            return self.__structural_parameters['gini']
        self.__set_field__('gini')
        seg_field = self.__field['gini']['detect']
        cal_field = self.__field['gini']['truncate']
        if seg_field not in self.__flag['get_gd']:
            self.__get_galaxy_data__(field=seg_field)
        if cal_field not in self.__flag['get_gd']:
            self.__get_galaxy_data__(field=cal_field)
        if cal_field not in self.__flag['get_seg']:
            self.__get_segmentation_map__(detect_field=seg_field, truncate_field=cal_field)
        _F = []
        gl = self.__galaxy
        radius_times = float(cal_field.replace('pr', ''))
        ptr = self.__petrosianRadius
        cty = ctx = ptr*radius_times
        for y in np.arange(2*radius_times*ptr+1):
            for x in np.arange(2*radius_times*ptr+1):
                dist = gl.cxx*(x-ctx)**2+gl.cyy*(y-cty)**2+gl.cxy*(x-ctx)*(y-cty)-3.5**2
                if self.__data['segmentation'][cal_field][y][x] and dist <= 0:
                    _F.append(self.__data['galaxy'][cal_field][y][x])
        n = len(_F)
        _F = np.array(sorted(_F))
        diff = np.array([2*(l+1)-n-1 for l in range(n)])
        self.__structural_parameters['gini'] = np.sum(diff * abs(_F)) / (abs(_F.mean()) * n * (n - 1))
        self.__flag['cal_g'] = True
        return self.__structural_parameters['gini']

    @property
    def moment_parameter(self):
        if self.__flag['cal_m']:
            return self.__structural_parameters['moment']
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        field = '2pr'
        times = float(field.replace('pr', ''))
        if field not in self.__flag['get_seg']:
            self.__get_segmentation_map__(detect_field=field, truncate_field=field)
        moment = np.copy(self.__data['galaxy'][field])
        gl = self.__galaxy
        ptr = self.__petrosianRadius*times
        seg = self.__data['segmentation'][field][gl.y][gl.x]
        sf = 0
        sm = 0
        for y in np.arange(2*ptr+1):
            for x in np.arange(2*ptr+1):
                moment[y][x] *= (y-gl.y)**2+(x-gl.x)**2
                if self.__data['segmentation'][field][y][x] == seg:
                    sf += self.__data['galaxy'][field][y][x]
                    sm += moment[y][x]
        arg = np.argsort(np.copy(self.__data['galaxy'][field]).flatten())[::-1]
        n = len(arg)
        f = 0
        m = 0
        for i in range(n):
            y = arg[i] // (2*ptr+1)
            x = arg[i] % (2*ptr+1)
            if self.__data['segmentation'][field][y][x] == seg:
                f += self.__data['galaxy'][field][y][x]
                m += moment[y][x]
                if f < sf*0.2:
                    self.__data['segmentation'][field][y][x] = -1
                    self.__structural_parameters['moment'] = m/sm
                    self.__flag['cal_m'] = True
        # if op.exists('tmp.fits'):
        #     subprocess.call('rm tmp.fits', shell=True, executable=self.__sys['shell'])
        # ft.writeto('tmp.fits', self.__data['segmentation'][field])
        return self.__structural_parameters['moment']

    @property
    def asymmetry_parameter(self):
        # ignore the influence of the background
        if self.__flag['cal_a']:
            return self.__structural_parameters['asymmetry']
        self.__set_field__('background')
        bg_field = self.__field['background']['detect']
        cal_field = self.__field['background']['truncate']
        self.__eliminate_pollutions__(detect_field=bg_field, deal_field=cal_field)
        gl = self.__galaxy
        ptr = self.__petrosianRadius
        radius_times = float(cal_field.replace('pr', ''))
        _I = np.copy(self.__data['galaxy'][bg_field][gl.y-ptr*radius_times:gl.y+ptr*radius_times+1,
                                                     gl.x-ptr*radius_times:gl.x+ptr*radius_times+1])

        _I180 = np.rot90(_I, 2)
        self.__structural_parameters['asymmetry'] = np.sum(abs(_I-_I180))/np.sum(abs(_I))
        self.__flag['cal_a'] = True
        return self.__structural_parameters['asymmetry']

    @property
    def concentration_parameter(self):
        if self.__flag['cal_c']:
            return self.__structural_parameters['concentration']
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        if not self.__flag['get_bg']:
            self.__get_background_map__(detect_field='sky')
        rms = self.__background['rms']
        n = len(self.__meanSurfaceBrightness)
        self.__circleApertureFlux = np.array([(2*(r+1)-1)**2 for r in range(n)])*self.__meanSurfaceBrightness
        try:
            k = float(np.argwhere(self.__surfaceBrightness < 2*rms)[0])
        except IndexError:
            k = float(np.argwhere(self.__surfaceBrightness < 3*rms)[0])
        r = min(n-1, self.__petrosianRadius*1.5, k)
        self.__structural_parameters['concentration'] = self.__circleApertureFlux[0.3*r]/self.__circleApertureFlux[r]
        self.__flag['cal_c'] = True
        return self.__structural_parameters['concentration']

    @property
    def structural_parameters(self):
        if not self.__flag['get_sp']:
            self.__structural_parameters['gini'] = float(self.gini_parameter)
            self.__structural_parameters['moment'] = float(self.moment_parameter)
            self.__structural_parameters['asymmetry'] = float(self.asymmetry_parameter)
            self.__structural_parameters['concentration'] = float(self.concentration_parameter)
            self.__flag['get_sp'] = True
        return json.dumps(self.__structural_parameters, indent=4, sort_keys=True)

    @property
    def half_light_radius_in_pixels(self):
        if not self.__flag['get_gd']:
            self.__get_galaxy_data__()
        n = len(self.__meanSurfaceBrightness)
        f = np.copy(self.__meanSurfaceBrightness)*[(2*r+1.0)**2 for r in range(n)]
        return float(np.argwhere(f > np.sum(self.__data['galaxy']['1pr'])*0.5)[0])

    # core functions
    # @log
    def __adjust_centroid_coordinate__(self):
        cty, ctx = self.__centroid['pix'][0], self.__centroid['pix'][1]
        core = self.__data['sky'][cty-10:cty+11, ctx-10:ctx+11].flatten()
        index = np.argmax(core)
        y = index // 21
        x = index % 21
        self.__centroid['pix'][0] += y-10
        self.__centroid['pix'][1] += x-10
        return

    # @log
    def __set_field__(self, item):
        if item == 'gini':
            if 'gini' not in self.__field:
                self.__field['gini'] = {
                    'detect': 'sky',
                    'truncate': '1.5pr'
                }
            else:
                if 'detect' not in self.__field['gini']:
                    self.__field['gini']['detect'] = 'sky'
                if 'truncate' not in self.__field['gini']:
                    self.__field['gini']['truncate'] = '1.5pr'
        elif item == 'background':
            if 'background' not in self.__field:
                self.__field['background'] = {
                    'detect': 'sky',
                    'truncate': '1.5pr'
                }
            else:
                if 'detect' not in self.__field['background']:
                    self.__field['background']['detect'] = 'sky'
                if 'truncate' not in self.__field['background']:
                    self.__field['background']['truncate'] = '1.5pr'
        return

    # @log
    def __get_petrosian_radius__(self):
        if self.__flag['get_pr']:
            return
        _rightDistance = np.array(self.__data['sky'].shape)-np.array(self.__centroid['pix'])
        _leftDistance = np.array(self.__centroid['pix'])
        _boxSize = int(np.min([_rightDistance, _leftDistance, [300, 300]]))
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
                    print(r)
                    self.__meanSurfaceBrightness[r + 1] += self.__meanSurfaceBrightness[r]
                self.__surfaceBrightness[r] /= 8*r
                self.__meanSurfaceBrightness[r] /= (2*(r+1)-1)**2
        eta = self.__surfaceBrightness/self.__meanSurfaceBrightness
        try:
            self.__petrosianRadius = float(np.argwhere(eta < 0.2)[0])
        except IndexError:
            self.__petrosianRadius = float(np.argwhere(eta < 0.25)[0])
        finally:
            self.__flag['get_pr'] = True
        return

    # @log
    def __get_galaxy_data__(self, field='sky'):
        if field in self.__flag['get_gd']:
            return
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        if field is 'sky':
            self.__data['galaxy'][field] = np.copy(self.__data['sky'])
        else:
            ct = self.__centroid['pix']
            times = float(field.replace('pr', ''))
            ptr = self.__petrosianRadius*times
            self.__data['galaxy'][field] = np.copy(self.__data['sky'][ct[0]-ptr:ct[0]+ptr+1, ct[1]-ptr:ct[1]+ptr+1])
        self.__flag['get_gd'][field] = True
        return

    # @log
    def __get_background_map__(self, detect_field='sky', truncate_field='1.5pr', ignore_flag=False):
        if truncate_field in self.__flag['get_bg'] and ignore_flag is False:
            return
        if detect_field not in self.__flag['get_bg']:
            self.__get_galaxy_data__(detect_field)
        if op.exists(self.__file['detect']):
            subprocess.call('rm %s' % self.__file['detect'], shell=True, executable=self.__sys['shell'])
        ft.writeto(self.__file['detect'], self.__data['galaxy'][detect_field])
        conf = '-CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s' % self.__file['background']
        _SExtractorProcess = subprocess.Popen('%s %s %s' % (self.__sys['sex'], self.__file['detect'], conf),
                                              shell=True, executable=self.__sys['shell'],
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _SExtractorProcess.wait()
        self.__SExtractorOutput = _SExtractorProcess.stdout.readlines()
        _backgroundIndex = self.__SExtractorOutput.index(b'\x1b[1M> Scanning image\n') - 1
        _backgroundInformation = self.__SExtractorOutput[_backgroundIndex].split()
        self.__background = {
            'mean': float(_backgroundInformation[2]),
            'rms': float(_backgroundInformation[4]),
            'threshold': float(_backgroundInformation[7]),
        }
        self.__get_catalog__(field=detect_field)
        with ft.open(self.__file['background']) as bg:
            if truncate_field == 'sky':
                self.__data['background'][truncate_field] = np.copy(bg[0].data)
            else:
                cty, ctx = self.__galaxy.y, self.__galaxy.x
                truncate_times = float(truncate_field.replace('pr', ''))
                ptr = self.__petrosianRadius
                self.__data['background'][truncate_field] = np.copy(bg[0].data[cty - ptr * truncate_times:cty + ptr * truncate_times + 1, ctx - ptr * truncate_times:ctx + ptr * truncate_times + 1])
            self.__flag['get_bg'][truncate_field] = True
        return

    # @log
    def __get_segmentation_map__(self, detect_field='sky', truncate_field='3pr', ignore_flag=False):
        if truncate_field in self.__flag['get_seg'] and ignore_flag is False:
            return
        if detect_field not in self.__flag['get_gd']:
            self.__get_galaxy_data__(detect_field)
        if op.exists(self.__file['detect']):
            subprocess.call('rm %s' % self.__file['detect'], shell=True, executable=self.__sys['shell'])
        ft.writeto(self.__file['detect'], self.__data['galaxy'][detect_field])
        conf = '-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME %s' % self.__file['segmentation']
        _SExtractorProcess = subprocess.Popen('%s %s %s' % (self.__sys['sex'], self.__file['detect'], conf),
                                              shell=True, executable=self.__sys['shell'])
        _SExtractorProcess.wait()
        self.__get_catalog__(field=detect_field)
        with ft.open(self.__file['segmentation']) as seg:
            if truncate_field == 'sky':
                self.__data['segmentation'][truncate_field] = np.copy(seg[0].data)
            else:
                cty, ctx = self.__galaxy.y, self.__galaxy.x
                truncate_times = float(truncate_field.replace('pr', ''))
                ptr = self.__petrosianRadius
                self.__data['segmentation'][truncate_field] = np.copy(seg[0].data[cty-ptr*truncate_times:cty+ptr*truncate_times+1, ctx-ptr*truncate_times:ctx+ptr*truncate_times+1])
            self.__flag['get_seg'][truncate_field] = True
        return

    # @log
    def __get_catalog__(self, field='sky'):
        if field is 'sky':
            cty, ctx = self.__centroid['pix']
        else:
            radius_times = float(field.replace('pr', ''))
            cty = ctx = self.__petrosianRadius*radius_times
        gls = pd.read_table(self.__file['catalog'], header=None, sep='\s+', names=['m', 'x', 'y', 'cxx', 'cyy', 'cxy'])
        gls['dist'] = abs(gls.y - cty) + abs(gls.x - ctx)
        gls = gls.sort_values(by=['dist'])
        gls.index = range(len(gls))
        self.__galaxy = gls.ix[0]
        self.__galaxy.y, self.__galaxy.x = cty, ctx
        return

    # @log
    def __eliminate_pollutions__(self, detect_field='10pr', deal_field='1.5pr'):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        if detect_field not in self.__flag['get_bg']:
            self.__get_background_map__(detect_field=detect_field, truncate_field=detect_field)
        if detect_field not in self.__flag['get_seg']:
            self.__get_segmentation_map__(detect_field=detect_field, truncate_field=detect_field)
        self.__get_catalog__(field=detect_field)
        ptr = self.__petrosianRadius
        radius_times = float(deal_field.replace('pr', ''))
        gl = self.__galaxy
        seg = self.__data['segmentation'][detect_field][gl.y][gl.x]
        rms = self.__background['rms']
        for y in np.arange(gl.y-ptr*radius_times, gl.y+ptr*radius_times+1):
            for x in np.arange(gl.x-ptr*radius_times, gl.x+ptr*radius_times+1):
                value = self.__data['segmentation'][detect_field][y][x]
                if value > seg:
                    self.__data['galaxy'][detect_field][y][x] = random.uniform(rms, rms*3)
        if op.exists(self.__file['tmp']):
            subprocess.call('rm %s' % self.__file['tmp'], shell=True, executable=self.__sys['shell'])
        # ft.writeto(self.__file['tmp'], self.__data['galaxy'][detect_field][gl.y-ptr*radius_times:gl.y+ptr*radius_times+1,
        #                                                                    gl.x - ptr * radius_times:gl.x + ptr * radius_times + 1])
        # subprocess.Popen('%s -scale mode zscale -zoom 2 %s' % (self.__sys['ds9'], self.__file['tmp']), shell=True, executable=self.__sys['shell'])
        return

    # visualizing methods
    def show_initial_image(self):
        subprocess.Popen('%s -scale mode zscale -zoom 0.25 %s' % (self.__sys['ds9'], self.__file['sky']),
                         shell=True, executable=self.__sys['shell'])
        return

    def show_eta_curve(self):
        if not self.__flag['get_pr']:
            self.__get_petrosian_radius__()
        sns.tsplot(self.__surfaceBrightness/self.__meanSurfaceBrightness)
        plt.show()
        return

    def show_galaxy_image(self, field='1pr', path='./', name='galaxy.fits', rotated=False):
        if field not in self.__flag['get_gd']:
            self.__get_galaxy_data__(field=field)
        if rotated is True:
            if op.exists('rotated_' + name):
                subprocess.call('rm %s' % path+'rotated_'+name, shell=True, executable=self.__sys['shell'])
            ft.writeto('rotated_' + name, np.rot90(self.__data['galaxy'][field], 2))
            subprocess.call('mv %s %s' % ('rotated_'+name, path + 'rotated_' + name), shell=True, executable=self.__sys['shell'])
            subprocess.Popen('%s -scale mode zscale -zoom 2 %s' % (self.__sys['ds9'], path + 'rotated_' + name), shell=True,
                             executable=self.__sys['shell'])
        else:
            if op.exists(name):
                subprocess.call('rm %s' % path+name, shell=True, executable=self.__sys['shell'])
            ft.writeto(name, self.__data['galaxy'][field])
            subprocess.call('mv %s %s' % (name, path+name), shell=True, executable=self.__sys['shell'])
            subprocess.Popen('%s -scale mode zscale -zoom 2 %s' % (self.__sys['ds9'], path+name), shell=True, executable=self.__sys['shell'])
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
            subprocess.call('rm ' + self.__file['truncate'], shell=True, executable=self.__sys['shell'])
        ft.writeto(self.__file['truncate'], self.__data['truncate'])
        subprocess.call('mv %s %s' % (self.__file['truncate'], path + name), shell=True, executable=self.__sys['shell'])
        subprocess.Popen('%s -scale mode zscale -zoom 2 %s' % (self.__sys['ds9'], path + name), shell=True, executable=self.__sys['shell'])
        return

    def show_residual_image(self, field='2pr', path='./', name='residual.fits'):
        if field not in self.__flag['get_gd']:
            self.__get_galaxy_data__(field=field)
        if op.exists(path+name):
            subprocess.call('rm %s' % path + name, shell=True, executable=self.__sys['shell'])
        ft.writeto(path+name, np.copy(self.__data['galaxy'][field])-np.rot90(self.__data['galaxy'][field], 2))
        subprocess.call('mv %s %s' % (name, path + name), shell=True, executable=self.__sys['shell'])
        subprocess.Popen('%s -scale mode zscale -zoom 2 %s' % (self.__sys['ds9'], path + name), shell=True, executable=self.__sys['shell'])
        return


# @log
def test():

    warnings.filterwarnings('ignore')
    catalog = pd.read_csv('list.csv')

    # # Linux Ubuntu version
    # fits_directory = '/home/franky/Desktop/type1cut/'
    # shell = '/usr/bin/zsh'
    # ds9 = 'ds9'
    # sex = 'sextractor'

    # Mac OS version
    fits_directory = '/Users/franky/Desktop/type1cut/'
    shell = '/bin/zsh'
    ds9 = '~/bin/ds9'
    sex = '/sw/bin/sex'

    for i in range(5, 6):
        ctl = catalog.ix[i]
        name = ctl.NAME1+'_r.fits'
        ct = [ctl.RA1, ctl.DEC1]
        gl = Galaxy(fits_directory+name, ct, shell=shell, ds9=ds9, sextractor=sex,
                    field={'gini': {'detect': 'sky', 'truncate': '1.5pr'},
                           'background': {'detect': 'sky', 'truncate': '1.5pr'}})
        gl.__eliminate_pollutions__(detect_field='sky', deal_field='2pr')


# @log
def load():

    # # Linux Ubuntu version
    # data = pd.read_table('/home/franky/Desktop/check/COSMOS-mor-H.txt', sep=' ', index_col=0)
    # fits = '/home/franky/Desktop/check/sky.fits'
    # shell = '/usr/bin/zsh'
    # ds9 = 'ds9'
    # sex = 'sextractor'

    # Mac OS version
    data = pd.read_table('/Users/franky/Desktop/check/COSMOS-mor-H.txt', sep=' ', index_col=0)
    fits = '/Users/franky/Desktop/check/sky.fits'
    shell = '/bin/zsh'
    ds9 = '~/bin/ds9'
    sex = '/sw/bin/sex'

    gls = data[(data.HalfLightRadiusInPixels > 20) &
               (data.HalfLightRadiusInPixels < 25)]
    gls.index = range(len(gls))
    lst = [5,  6,  7,  9, 11, 12, 13, 15, 16, 17, 21, 25, 28, 29, 31, 32, 33,
           34, 42, 43, 45, 46, 47]
    x = []
    y = []
    ratio = []
    for i in lst[0:1]:
        sample = gls.ix[i]
        print(sample)
        gl = Galaxy(fits, [sample.Y_IMAGE, sample.X_IMAGE], centroid_mode='pix', shell=shell, ds9=ds9, sextractor=sex,
                    field={'gini': {'detect': 'sky', 'truncate': '1.5pr'},
                           'background': {'detect': '5pr', 'truncate': '1.5pr'}})
        gl.__eliminate_pollutions__()
        # print(gl.asymmetry_parameter)
        # gl.show_galaxy_image(field='1.5pr')
        # gl.show_galaxy_image(field='1.5pr', rotated=True)
        # gl.show_residual_image()
        # ratio.append(abs(gl.moment_parameter-sample.M20)/sample.M20)
        # x.append(gl.moment_parameter)
        # y.append(sample.M20)
    for i in range(len(ratio)):
        print(i, x[i], y[i], ratio[i])
    # sns.tsplot(x, color='r')
    # sns.tsplot(y, color='b')
    # plt.show()


if __name__ == '__main__':
    test()
    # load()
