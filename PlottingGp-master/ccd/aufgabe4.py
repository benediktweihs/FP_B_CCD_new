
from __future__ import unicode_literals
from uncertainties import unumpy
from uncertainties import *
import uncertainties
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import scipy.odr.odrpack as odrpack
from matplotlib import rc
from matplotlib import gridspec
rc('text', usetex=True)



def meanValue(uncertainArray):
    x = unumpy.nominal_values(uncertainArray)
    y = unumpy.std_devs(uncertainArray)

    g = [1/(i**2) for i in y]
    gewicht = 1/sum(g)
    partial = 0
    for i in range(len(y)):
        partial = partial + x[i]/(y[i]**2)
    nominal = gewicht * partial
    stdDev = np.sqrt(gewicht)
    return [nominal, stdDev]

sigma_ron=[12.8883,6.389016,4.518182,2.109611,1.124224,1.251086,-0.03628499,-0.00940241,-0.7051026,-0.4990073,-1.831167,-1.898964,-1.39846,-2.46305,-2.485171,-2.501839,-3.56378,-3.564269,-2.931819,-4.392106]
sigma_ron_err=[6.98959,6.92064,6.944537,6.918103,6.931074,6.946267,6.908369,6.937337,6.914836,6.952673,6.91996,6.931309,6.943927,6.922836,6.938537,6.946637,6.917567,6.928072,6.953143,6.917483]

sigma=unumpy.uarray(sigma_ron,sigma_ron_err)
temp=0
for i in sigma_ron:
    temp+=i*i
print(np.sqrt(temp/20))
print(meanValue(sigma))

pixel=40401
a4_Nadu_g=np.array([1.27E+03,2.52E+03,4.84E+03,7.16E+03,9.49E+03,1.18E+04,1.42E+04,1.77E+04,2.36E+04,3.55E+04,4.25E+04,4.84E+04,5.19E+04,5.43E+04,5.66E+04,5.91E+04,6.17E+04,6.37E+04,6.49E+04])
a4_std_g=np.array([2.65E+01,3.62E+01,5.02E+01,6.25E+01,7.46E+01,8.56E+01,9.65E+01,1.12E+02,1.38E+02,1.88E+02,2.16E+02,2.38E+02,2.52E+02,2.61E+02,2.66E+02,2.81E+02,2.47E+02,1.95E+02,3.80E+00])
a4_Nadu_g_err=np.sqrt(a4_Nadu_g)/np.sqrt(pixel)#a4_std_g ##sqrt(N)

a4_std_g_err=0.1 ##??
a4_t_g=np.array([0.5,1,2,3,4,5,6,7.5,10,15,18,20.5,22,23,24,25,26,27,28])


a4_Nadu_b=np.array([1.54E+03,3.05E+03,5.82E+03,8.63E+03,1.14E+04,1.42E+04,1.71E+04,2.13E+04,2.84E+04,4.26E+04,4.96E+04,5.38E+04,5.66E+04,5.95E+04,6.24E+04,6.46E+04,6.49E+04])
a4_std_b=np.array([2.86E+01,3.91E+01,5.37E+01,6.70E+01,7.88E+01,9.05E+01,1.02E+02,1.18E+02,1.42E+02,1.91E+02,2.12E+02,2.24E+02,2.31E+02,2.36E+02,1.78E+02,1.43E+02,1.84E+00])
a4_Nadu_b_err=np.sqrt(a4_Nadu_b)/np.sqrt(pixel)#a4_std_b/np.sqrt(pixel) ##sqrt(N)
a4_std_b_err=0.1#np.sqrt(pixel) ##??
a4_t_b=np.array([0.5,1,2,3,4,5,6,7.5,10,15,17.5,19,20,21,22,23,24])

Nadu_g=unumpy.uarray(a4_Nadu_g,a4_Nadu_g_err)
Nadu_b=unumpy.uarray(a4_Nadu_b,a4_Nadu_b_err)
a3_Nadu_g=Nadu_g[1]
a3_Nadu_b=Nadu_b[1]
a3_std_g=ufloat(a4_std_g[1],a4_std_g_err)
a3_std_b=ufloat(a4_std_b[1],a4_std_b_err)
gain_g=a3_Nadu_g/a3_std_g/a3_std_g
gain_b=a3_Nadu_b/a3_std_b/a3_std_b
temp_g=0
temp_b=0
for i in range(7,12+1): #blau ... zwischen 7.5 und 20 , grün ... zw. 7.5 und 22
    #print(a4_t_b[i])
    temp_g += Nadu_g[i]/a4_t_g[i]
    temp_b += Nadu_b[i]/a4_t_b[i]

Nadu_mean_g=temp_g/6
Nadu_mean_b=temp_b/6

print('Aufgabe3:')
print('gain grün =',gain_g)
print('gain blau =',gain_b)
print('Aufgabe4:')
print('Nadu mean grün =',temp_g/6)
print('Nadu mean blau =',temp_b/6)

y1_g = Nadu_g/a4_t_g
y1_b = Nadu_b/a4_t_b
y2_g = Nadu_g/a4_t_g/Nadu_mean_g
y2_b = Nadu_b/a4_t_b/Nadu_mean_b

y1_g_err = unumpy.std_devs(Nadu_g/a4_t_g)
y1_b_err = unumpy.std_devs(Nadu_b/a4_t_b)
y2_g_err = unumpy.std_devs(Nadu_g/a4_t_g/Nadu_mean_g)
y2_b_err = unumpy.std_devs(Nadu_b/a4_t_b/Nadu_mean_b)

fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=2,height_ratios= [1, 2])#width_ratios=[2, 1])
ax = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])

#fig = plt.figure(figsize=(8, 6))


#ax.set_xlabel(r'$t_{exp}$ in s')
ax.set_ylabel(r'$N^{ADU}/t_{exp}$')
#ax2=ax.twinx()
ax.plot(a4_t_g,a4_Nadu_g/a4_t_g,marker='o',linestyle='',markersize=3,label='grüner Filter',color='green')
ax.errorbar(a4_t_g,a4_Nadu_g/a4_t_g,yerr=y1_g_err,marker='o',linestyle='',linewidth=0.5,markersize=0,capsize=2,color='green')
ax.errorbar(a4_t_b,a4_Nadu_b/a4_t_b,yerr=y1_b_err,marker='o',linestyle='',linewidth=0.5,markersize=0,capsize=2,color='blue')
ax.plot(a4_t_b,a4_Nadu_b/a4_t_b,marker='o',linestyle='',markersize=3,label='blauer Filter',color='blue')
#plt.show()

#ax2 = fig.subplot(gs[1])
#ax2=figure.add_subplot(212)
ax2.set_xlabel(r'$t_{exp}$ in s')
ax2.set_ylabel(r'$N^{ADU}/(t_{exp} \cdot N^{ADU}_{mean})$')
#ax2=ax.twinx()
ax2.plot(a4_t_g,a4_Nadu_g/(a4_t_g*nominal_value(Nadu_mean_g)),marker='o',linestyle='',markersize=3,label='grüner Filter',color='green')
ax2.plot(a4_t_b,a4_Nadu_b/(a4_t_b*nominal_value(Nadu_mean_b)),marker='o',linestyle='',markersize=3,label='blauer Filter',color='blue')
ax2.errorbar(a4_t_g,a4_Nadu_g/(a4_t_g*nominal_value(Nadu_mean_g)),yerr=y2_g_err,marker='o',linestyle='',linewidth=0.5,markersize=0,capsize=2,color='green')
ax2.errorbar(a4_t_b,a4_Nadu_b/(a4_t_b*nominal_value(Nadu_mean_b)),yerr=y2_b_err,marker='o',linestyle='',linewidth=0.5,markersize=0,capsize=2,color='blue')
ax2.hlines(1,-1,30,color='black',linestyles='-.',linewidth=0.5)
ax2.set_xlim(-1,30)
plt.legend()
#ax2.set_aspect('equal')
plt.show()
plt.savefig('Aufgabe4.pdf')


