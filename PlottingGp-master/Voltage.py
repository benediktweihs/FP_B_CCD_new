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
data = convert("Oszi_U_front")
#print(data[0])

fast=4
slow=7
slice=300

time = np.array(data[0],dtype=float)
time=time[0::5]
Ufuse = np.array(data[2], dtype=float)
Ufuse=Ufuse[0::5]
Ubat = np.array(data[1], dtype=float)
Ubat=Ubat[0::5]


# Abbidlungen
fig, ax = plt.subplots()
ax.plot(time,Ubat, color = 'red', marker='o', linestyle='', markersize = 1,label=r'$U_{Bat}$')
ax.plot(time,Ufuse, color = 'blue', marker='o', linestyle='', markersize = 1,label=r'$U_{Fuse}$')
ax.plot(time,Ubat-Ufuse,color = 'green', marker='o', linestyle='',markersize = 1,label=r'$U_{ConductivePath}$')
ax.legend(loc='upper left',frameon=True)
ax.set(xlabel='Time (ms)',ylabel='Voltage (V)')
ax.grid(True,linestyle='--',linewidth = 0.3)
#plt.xlim(-70,2600)
plt.ylim(-0.2,5.5)



#plt.xscale('log')
#plt.title('test')

#plt.errorbar(X, Y, yerr=0.1, fmt='ko', linewidth=0.8, capsize=3, capthick=0.8, markersize=5)

i='Voltage'
plt.savefig("Graphen/Temperature_" + str(i) + ".png")

plt.show()