import galaxy
import warnings
import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')

# Mac OS version
type1_fits_directory = '/Users/franky/Desktop/type1cut/'
type2_fits_directory = '/Users/franky/Desktop/type2cut/'
shell = '/bin/zsh'
ds9 = '~/bin/ds9'
sex = '/sw/bin/sex'

# data catalog
catalog = pd.read_csv('list.csv')
catalog = catalog[catalog.Z1 < 0.05]
catalog.index = range(len(catalog))

# calculated flag
calculated_set1 = {}
calculated_set2 = {}

fout = open('datax.csv', 'w')
fout.write('NAME1,RA1,DEC1,Z1,PR1,G1,M1,NAME2,RA2,DEC2,Z2,PR2,G2,M2\n')
print(catalog)
for i in range(len(catalog)):
    print(i)
    ctl = catalog.ix[i]
    type1 = ctl.NAME1
    if type1 not in calculated_set1:
        ct1 = [ctl.RA1, ctl.DEC1]
        gl1 = galaxy.Galaxy(type1_fits_directory + type1 + '_r.fits', ct1, shell=shell, ds9=ds9, sextractor=sex,
                            field={'gini': {'detect': 'sky', 'truncate': '1.5pr'},
                                   'background': {'detect': 'sky', 'truncate': '1.5pr'}})
        catalog.at[i, 'G1'] = gl1.gini_parameter
        catalog.at[i, 'M1'] = gl1.moment_parameter
        # catalog.at[i, 'A1'] = gl1.asymmetry_parameter
        # catalog.at[i, 'C1'] = gl1.concentration_parameter
        catalog.at[i, 'PR1'] = gl1.petrosian_radius
        calculated_set1[type1] = i
    else:
        k = calculated_set1[type1]
        catalog.at[i, 'G1'] = catalog.at[k, 'G1']
        catalog.at[i, 'M1'] = catalog.at[k, 'M1']
        # catalog.at[i, 'A1'] = catalog.at[k, 'A1']
        # catalog.at[i, 'C1'] = catalog.at[k, 'C1']
        catalog.at[i, 'PR1'] = catalog.at[k, 'PR1']

    type2 = ctl.NAME2
    if type2 not in calculated_set2:
        ct2 = [ctl.RA2, ctl.DEC2]
        gl2 = galaxy.Galaxy(type2_fits_directory + type2 + '_r.fits', ct2, shell=shell, ds9=ds9, sextractor=sex,
                            field={'gini': {'detect': 'sky', 'truncate': '1.5pr'},
                                   'background': {'detect': 'sky', 'truncate': '1.5pr'}})
        catalog.at[i, 'G2'] = gl2.gini_parameter
        catalog.at[i, 'M2'] = gl2.moment_parameter
        # catalog.at[i, 'A2'] = gl2.asymmetry_parameter
        # catalog.at[i, 'C2'] = gl2.concentration_parameter
        catalog.at[i, 'PR2'] = gl2.petrosian_radius
        calculated_set2[type2] = i

    else:
        k = calculated_set2[type2]
        catalog.at[i, 'G2'] = catalog.at[k, 'G2']
        catalog.at[i, 'M2'] = catalog.at[k, 'M2']
        # catalog.at[i, 'A2'] = catalog.at[k, 'A2']
        # catalog.at[i, 'C2'] = catalog.at[k, 'C2']
        catalog.at[i, 'PR2'] = catalog.at[k, 'PR2']

    # tmp = catalog.ix[i].reindex(['NAME1', 'RA1', 'DEC1', 'Z1', 'PR1', 'G1', 'M1', 'A1', 'C1',
    #                              'NAME2', 'RA2', 'DEC2', 'Z2', 'PR2', 'G2', 'M2', 'A2', 'C2']).values

    tmp = catalog.ix[i].reindex(['NAME1', 'RA1', 'DEC1', 'Z1', 'PR1', 'G1', 'M1',
                                 'NAME2', 'RA2', 'DEC2', 'Z2', 'PR2', 'G2', 'M2']).values
    for j in range(14):
        if j < 13:
            fout.write(str(tmp[j])+',')
        else:
            fout.write(str(tmp[j])+'\n')
    fout.flush()
