from __future__ import unicode_literals
from uncertainties import *
from converter import *
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

data = convert("Wärmepumpe")
#datael=convert("Leistung")
dataf=convert("fehler")
#print(data[0])

m1=ufloat(4.36, 0.02)
m2=ufloat(4.28, 0.02)
m3=ufloat(0.36, 0.02)
t1=ufloat(304.7187, 0.03)
t2=ufloat(308.3056, 0.03)
t3=ufloat(281.0657, 0.03)
t4=ufloat(277.2301, 0.03)
Hk=ufloat(175.04, 0.01)
Hh=ufloat(196.77,0.01)

th=t2-t1
tk=t3-t4
mh=m1-m3
mk=m2-m3

msh= (4182*th*mh)/(240.0 * Hh * 1000.0)
print(msh*60)
msk= (4182*tk*mk)/(240.0 * Hk * 1000.0)
print(msk*60)
print(mh)
print(mk)
print(tk)
print(th)

#fehlerauswertung
timef = np.array(dataf[0],dtype=float)  #ms
Channel0f = np.array(dataf[8],dtype=float)  #Kelvin
#print(np.std(Channel0f))    #ca 0.03 Kelvin
#timef=timef[0:500:]
#Channel0f=Channel0f[0:500:]

#temperatur auswertung
time = np.array(data[0],dtype=float)  #ms
Channel2 = np.array(data[3],dtype=float)  #Kelvin
time=time[0::60]
Channel2 =Channel2[0::60]

Channel4 = np.array(data[5],dtype=float)  #Kelvin
time=time[0::60]
Channel4 =Channel4[0::60]

#print(time/60)
#print(Channel4)

#time1 = np.array(data[0],dtype=float)
#time=time[0::5]
#Ufuse = np.array(data[2], dtype=float)
#Ufuse=Ufuse[0::5]
#Ubat = np.array(data[1], dtype=float)
#Ubat=Ubat[0::5]


###################
#n*lambda/2  mbar
p=[]
pu=940
p0=[680,650,620,590,560,520,500,470,430,400,370,340,290,240,210,160,120,90,60]

for i in range(len(p0)):
    p.append(pu-p0[i])
















#################



# Abbidlungen
fig, ax = plt.subplots()
#ax.plot(timef,Channel0f, color = 'red', marker='o', linestyle='', markersize = 1,label=r'$U_{Bat}$')
#ax.plot(time,Ufuse, color = 'blue', marker='o', linestyle='', markersize = 1,label=r'$U_{Fuse}$')
#ax.plot(time,Ubat-Ufuse,color = 'green', marker='o', linestyle='',markersize = 1,label=r'$U_{ConductivePath}$')
ax.legend(loc='upper left',frameon=True)
#ax.set(xlabel='Time (ms)',ylabel='Voltage (V)')
ax.grid(True,linestyle='--',linewidth = 0.3)
#plt.xlim(-70,2600)
#plt.ylim(-0.2,5.5)

#plt.hist(Channel0f, bins=5)


#plt.xscale('log')
#plt.title('test')

#plt.errorbar(X, Y, yerr=0.1, fmt='ko', linewidth=0.8, capsize=3, capthick=0.8, markersize=5)

#i='Voltage'
#plt.savefig("Graphen/Temperature_" + str(i) + ".png")

#plt.show()