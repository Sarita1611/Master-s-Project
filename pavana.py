# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:39:49 2023

@author: sana kodakeri
"""

from astropy.io import fits
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(15,7))

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
fig, ax = plt.subplots()
w16,f16=np.loadtxt('comb.txt', usecols=(0,1), unpack= True)
#w1,f1=np.loadtxt('24apr_1c.txt', usecols=(0,2), unpack= True)
#w,f=np.loadtxt('comb.txt', usecols=(0,1), unpack=True)
#plt.plot(w,f/7.461697E-13,'g',label='ncomb')
#plt.plot(w1,f1,'r',label='gr7')

plt.plot(w16,f16,'b',label='gr7 +gr8')

plt.legend()

#plt.xlim(3630,7900)
#plt.ylim(5,8)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_major_formatter('{x:.0f}')
plt.plot(w16,f16/6.891974E-12,'b',label='16_april')
ax.xaxis.set_minor_locator(MultipleLocator(100))

ax.legend()
plt.xlabel('Wavelength(A)')
plt.ylabel('Relative Flux')
#plt.xlim(6500,6700)
#plt.savefig('norm by hbeta_0.6_24apr_comp.jpg')
fn16=f16/5.541698E-13
fn16
myFile = open('test', 'w+')

np.savetxt(myFile, fn16)
myFile.close()
plt.plot(fn16)
plt.show()
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
w16n,f16n=np.loadtxt('16april_0.4_norm', usecols=(0,2), unpack= True)
#plt.plot(w16n,f16n,'k')
fig, ax = plt.subplots()
ax.plot(w16n,f16n,'k',label='16_april_0.4')


ax.yaxis.set_major_locator(MultipleLocator(500))

ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_major_formatter('{x:.0f}')
plt.plot(w16,f16/6.891974E-12,'b',label='16_april')
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.legend()
plt.xlabel('Wavelength(A)')
plt.ylabel('Relative Flux')

#plt.savefig('norm by 4861_0.4_greater than 3900.jpg')
plt.rcParams['figure.figsize'] = (15, 9)
plt.rcParams['font.size'] = 17                                     #set the size of the font numbers       
plt.rcParams['font.family'] = 'fantasy'                            #choose the style of the numbers
plt.rcParams['axes.labelsize'] = 20                                #set the size of the word axes
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']        #set the size of the label x
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']        #set the size of the label y
plt.rcParams['xtick.major.size'] = 8                               #set the length of the x tick
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.width'] = 2                              #set the width of the x tick 
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 7                               #set the length of the y tick
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.width'] = 2                              #set the width of the y tick
plt.rcParams['ytick.minor.width'] = 1.2
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.loc'] = 'upper left'
plt.rcParams['axes.linewidth'] = 1.5
import pandas as pd
w=np.loadtxt('16april_0.4_norm',usecols=(0,2))
c= w[:1]>=3900
c1=w[:1]<=4100
c2= w[:1][c1]
len(c2)
np.shape(w)
day= (6,18,19,21,23,45,57,65,68,140,169,174,239)
halpha=(88,63,62,60,59.6,105,108,108.1,108,94.5,90,91,89)
hbeta=(70,45.5,44.7,47.4,47.5,113,115)
day2= (6,18,19,21,23,45,57)
#plt.plot(day,halpha,'bo')
#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
fig, ax = plt.subplots()
plt.scatter(day,halpha, marker ="^",s=100,label='H-alpha')
plt.scatter(day2,hbeta, marker ="^",s=100,label='H-beta')

ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(2))

ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_major_formatter('{x:.0f}')
#plt.plot(w16,f16/6.891974E-12,'b',label='16_april')
ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.ylabel('FWHM')
plt.xlabel('Days after outburst')
plt.legend()
plt.savefig('fwhm.jpg')
from scipy.optimize import curve_fit
import scipy
scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  day,  halpha,  p0=(4, 0.1))

