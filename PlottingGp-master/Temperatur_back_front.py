from __future__ import unicode_literals
from uncertainties import *
from converterNew import *
from uncertainties import unumpy
from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

##matplotlib.rcParams['text.usetex'] = True
##matplotlib.rcParams['text.latex.unicode'] = True
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#alle Graphen werden in Graphen gespeichert
if not os.path.exists("Graphen"):
    os.mkdir("Graphen")

#data enthält sämtliche Information
#data = np.array(convert("messwerteText"))

#data = convert("test")
data = convert("Temperatur_back_front")
#print(data[0])

fast=4
slow=7
slice=300

time1 = np.array(data[0],dtype=float)
time1 = np.concatenate((time1[0:slice:fast],time1[slice:1971:slow]))
#time1=time1[0::2]
time2 = np.array(data[3],dtype=float)
time2 = np.concatenate((time2[0:slice:fast],time2[slice::slow]))
#time2=time2[0::2]
T1a = np.array(data[2], dtype=float)
T1a = np.concatenate((T1a[0:slice:fast],T1a[slice:1971:slow]))
#T1a=T1a[0::2]
T1p = np.array(data[1], dtype=float)
T1p = np.concatenate((T1p[0:slice:fast],T1p[slice:1971:slow]))
#T1p=T1p[0::2]
T2a = np.array(data[5], dtype=float)
T2a = np.concatenate((T2a[0:slice:fast],T2a[slice::slow]))
#T2a=T2a[0::2]
T2p = np.array(data[4], dtype=float)
T2p = np.concatenate((T2p[0:slice:fast],T2p[slice::slow]))
#T2p=T2p[0::2]
#x = x[0::100]

#y = y[0::100]


# Abbidlungen
fig, ax = plt.subplots()
ax.plot(time1,T1p, color = 'blue', marker='o', linestyle='', markersize = 1)
ax.plot(time1,T1a,'b--',linewidth = 0.6)
ax.plot(time2,T2p,'ro',markersize = 1)
ax.plot(time2,T2a,'r--',linewidth = 0.6)
ax.set(xlabel='Time (s)',ylabel='Temperature (K)')
ax.grid(True,linestyle='--',linewidth = 0.3)
#ax.legend(loc='upper left',frameon=True)

plt.xlim(-70,2600)
plt.ylim(-1,15.9)
plt.annotate('a', xy=(400,14), fontsize=14)
plt.annotate('b', xy=(400,10.5), fontsize=14)
plt.annotate('c', xy=(400,8.3), fontsize=14)
plt.annotate('d', xy=(400,6.3), fontsize=14)

#plt.annotate('a', xy=(2100,14), fontsize=14)
#plt.annotate('b', xy=(2100,11), fontsize=14)
#plt.annotate('c', xy=(2125,9), fontsize=14)
#plt.annotate('d', xy=(2150,6.5), fontsize=14)

#plt.xscale('log')
#plt.title('test')

#plt.plot(pointX, pointY, 'go', markersize=5)
#plt.plot(lineX, lineY, 'r--', linewidth=0.8)
#plt.errorbar(X, Y, yerr=0.1, fmt='ko', linewidth=0.8, capsize=3, capthick=0.8, markersize=5)
#plt.xlabel(r'$\textbf{Time } (ms)$')
#plt.ylabel(r'$\textbf{Temperature } (K)$')
i='back_front'
plt.savefig("Graphen/Temperature_" + str(i) + ".png")

plt.show()