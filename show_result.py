import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# R Band figure

# get data
data = pd.read_csv('data.csv')
data2 = pd.read_csv('data2.csv')
# picture settings
fr, ar = plt.subplots(2, 2, figsize=(16, 8), sharex=False, sharey=False)
fr.suptitle('R Band Differences')

# Gini Coefficient
rg = sns.distplot(data[data.G1 > 0.4]['G1'].values, bins=100, color='r', ax=ar[0, 0], hist=False, kde=True, label='type1')
rg.set(xlabel='Gini Coefficient', ylabel='Gaussian Kernel Density')
sns.distplot(data[data.G2 > 0.4]['G2'].values, bins=100, color='b', ax=ar[0, 0], hist=False, kde=True, label='type2')


#  Concentration Index
rc = sns.distplot(data['C1'].values, bins=100, color='m', ax=ar[0, 1], hist=False, kde=True, label='type1')
rc.set(xlabel='Concentration Index')
sns.distplot(data['C2'].values, bins=100, color='y', ax=ar[0, 1], hist=False, kde=True, label='type2')

#  Moment Index
rm = sns.distplot(data[data.M1 < 0.02]['M1'].values, bins=100, color='g', ax=ar[1, 0], hist=False, kde=True, label='type1')
rm.set(xlabel='Moment Index', ylabel='Gaussian Kernel Density')
sns.distplot(data[data.M2 < 0.02]['M2'].values, bins=100, color='c', ax=ar[1, 0], hist=False, kde=True, label='type2')
rm2 = sns.distplot(data2[data2.M1 < 0.12]['M1'].values, bins=100, color='g', ax=ar[1, 1], hist=False, kde=True, label='type1')
rm2.set(xlabel='Moment Index (agn removed)', ylabel='Gaussian Kernel Density')
sns.distplot(data2[data2.M2 < 0.12]['M2'].values, bins=100, color='c', ax=ar[1, 1], hist=False, kde=True, label='type2')
# rm = sns.distplot(np.log10(data['M1'].values), bins=100, color='g', ax=ar[1, 0], hist=False, kde=True, label='type1')
# rm.set(xlabel='Moment Index', ylabel='Gaussian Kernel Density')
# sns.distplot(np.log10(data['M2'].values), bins=100, color='c', ax=ar[1, 0], hist=False, kde=True, label='type2')
# rm2 = sns.distplot(np.log10(data2['M1'].values), bins=100, color='g', ax=ar[1, 1], hist=False, kde=True, label='type1')
# rm2.set(xlabel='Moment Index (agn removed)', ylabel='Gaussian Kernel Density')
# sns.distplot(np.log10(data2['M2'].values), bins=100, color='c', ax=ar[1, 1], hist=False, kde=True, label='type2')

# #  Asymmetry Index
# rm = sns.distplot(data[data.A1 < 0.6]['A1'].values, bins=100, color='brown', ax=ar[1, 1], hist=False, kde=True, label='type1')
# rm.set(xlabel='Asymmetry Index')
# sns.distplot(data[data.A2 < 0.6]['A2'].values, bins=100, color='hotpink', ax=ar[1, 1], hist=False, kde=True, label='type2')

plt.show()
