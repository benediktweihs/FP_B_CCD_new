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

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# alle Graphen werden in Graphen gespeichert
if not os.path.exists("Graphen"):
    os.mkdir("Graphen")

# data enthält sämtliche Information
# data = np.array(convert("messwerteText"))

# data = convert("test")
data = convert("Temperatur_front")
eldata = convert("UI_front")
# print(data[0])

fast = 2
slow = 5
slice = 300

# retrieve data temperature
time1 = np.array(data[0], dtype=float)
time1 = np.concatenate((time1[0:slice:fast], time1[slice:2225:slow]))
# time1=time1[0::2]
time2 = np.array(data[2], dtype=float)
time2 = np.concatenate((time2[0:slice:fast], time2[slice::slow]))
# time2=time2[0::2]
T1p = np.array(data[1], dtype=float)
T1p = np.concatenate((T1p[0:slice:fast], T1p[slice:2225:slow]))
# T1p=T1p[0::2]
T2p = np.array(data[3], dtype=float)
T2p = np.concatenate((T2p[0:slice:fast], T2p[slice::slow]))
# T2p=T2p[0::2]
# x = x[0::100]

# y = y[0::100]

# retrieve data U_I
time3 = np.array(eldata[0], dtype=float)
# time2 = np.concatenate((time1[0:slice:fast],time1[slice:2225:slow]))
# time1=time1[0::2]
# time2 = np.array(data[2],dtype=float)
# time2 = np.concatenate((time2[0:slice:fast],time2[slice::slow]))
# time2=time2[0::2]
Current = np.array(eldata[1], dtype=float)
# T1p = np.concatenate((T1p[0:slice:fast],T1p[slice:2225:slow]))
# T1p=T1p[0::2]
Vbat = np.array(eldata[2], dtype=float)
V2 = np.array(eldata[3], dtype=float)
# T2p = np.concatenate((T2p[0:slice:fast],T2p[slice::slow]))
# T2p=T2p[0::2]

Currentu = []
Ubatu = []
U2u = []
Ufuseu = []
Rfuseu = []
Pfuseu = []
time3u = []

fit=[]

for i in range(len(time3)):
    time3u.append(ufloat(time3[i], 0.0))
    Currentu.append(ufloat(Current[i], 0.0001))
    Ubatu.append(ufloat(Vbat[i], 0.015))    ##berechnete standartabweichung
    U2u.append(ufloat(V2[i], 0.005))        ##max auflösungsvermögen bei 5mV
    Ufuseu.append(Ubatu[i] - U2u[i])

    #fit.append(3.94385322+time3*(-2.69122910e-07))

    if (i<=2):
        Rfuseu.append(0)
        Pfuseu.append(0)
    if (i > 2):
        Rfuseu.append(Ufuseu[i] / Currentu[i])
        Pfuseu.append(Ufuseu[i] * Currentu[i])

print(unumpy.std_devs(Rfuseu))
print(unumpy.std_devs(Pfuseu))

# fig, ax2 = plt.subplots()
# plt.plot(unumpy.nominal_values(time3u),unumpy.nominal_values(Ubatu))


# Abbidlungen
X=unumpy.nominal_values(time3u)
Ubat=unumpy.nominal_values(Ubatu)
UbatErr=unumpy.std_devs(Ubatu)
Ufuse=unumpy.nominal_values(Ufuseu)
UfuseErr=unumpy.std_devs(Ufuseu)
Pfuse=unumpy.nominal_values(Pfuseu)
PfuseErr=unumpy.std_devs(Pfuseu)
Rfuse=unumpy.nominal_values(Rfuseu)
RfuseErr=unumpy.std_devs(Rfuseu)
Current=unumpy.nominal_values(Currentu)
CurrentErr=unumpy.std_devs(Currentu)

#plt.subplot(311)
fig,axarr = plt.subplots(3)
#plt.figure(1)       #Strom und Spannung


axarr[0].set_xlim(-20, 290)
#plt.ylim(-1, 7)
ax2=axarr[0].twinx()
p1, = axarr[0].plot(X, Ubat, color='black', marker='x', linestyle='',markersize=3, label=r'$U_{Bat}$')
axarr[0].errorbar(X,Ubat,yerr=UbatErr, color='black', linestyle='', linewidth=1, capsize=1, capthick=1)
p2, = axarr[0].plot(X, Ufuse, color='green', marker='x', linestyle='',markersize=3, label=r'$U_{Fuse}$')
axarr[0].errorbar(X,Ufuse,yerr=UfuseErr, color='green', linestyle='', linewidth=1, capsize=1, capthick=1)
p3, = ax2.plot(X, Current*1000, color='red', marker='x', linestyle='',markersize=3, label=r'$I$')
ax2.errorbar(X,Current*1000,yerr=CurrentErr*1000, color='red', linestyle='', linewidth=1, capsize=1, capthick=1)

axarr[0].set_xlabel("Time (ms)")
axarr[0].set_ylabel("Voltage (V)")
ax2.set_ylabel("Current (mA)")

#axarr[0].yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p3.get_color())

#axarr[1].tick_params(axis='y', colors=p1.get_color())
ax2.tick_params(axis='y', colors=p3.get_color())

lines=[p1,p2,p3]
axarr[0].legend(lines, [l.get_label() for l in lines], loc='lower right', frameon=True)
#axarr[0].legend(loc='lower right', frameon=True)

#fig,ax1 = plt.subplots(312)
#plt.subplot(312)

ax2=axarr[1].twinx()
axarr[1].set_xlim(-20, 290)
p1, = axarr[1].plot(X, Pfuse, color='red', marker='x', linestyle='',markersize=2, label=r'$P_{Fuse}$')
axarr[1].errorbar(X,Pfuse,yerr=PfuseErr, color='red', linestyle='', linewidth=1, capsize=1, capthick=1)
p2, = ax2.plot(X, Rfuse, color='blue', marker='x', linestyle='',markersize=2, label=r'$R_{Fuse}$')
ax2.errorbar(X,Rfuse,yerr=RfuseErr, color='blue', linestyle='', linewidth=1, capsize=1, capthick=1)

axarr[1].set_xlabel("Time (ms)")
axarr[1].set_ylabel("Power (W)")
ax2.set_ylabel("Resistsance (Ohm)")

axarr[1].yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p2.get_color())

axarr[1].tick_params(axis='y', colors=p1.get_color())
ax2.tick_params(axis='y', colors=p2.get_color())

lines=[p1,p2]
axarr[1].legend(lines, [l.get_label() for l in lines], loc='lower right', frameon=True)
#ax2.legend(loc='lower right', frameon=True)


#plt.subplot(313)
X=X[0::10]
Ubat=Ubat[0::10]
#fit=fit[0::10]
axarr[2].plot( X, Ubat, color='black', marker='o', linestyle='',markersize=0.5, label=r'$U_{Bat}$')


#axarr[2].plot( X, 3.95094740+X*(-2.78023652e-07), color='blue', marker='', linestyle='--',markersize=0.0, label=r'$linfit: U_{Bat}$')

axarr[2].plot( X, 3.94385322+X*(-2.69122910e-07), color='red', marker='', linestyle='--',markersize=0.0, label=r'$linfit: U_{Bat}$')
axarr[2].set_xlabel("Time (ms)")
axarr[2].set_ylabel("Voltage (V)")
axarr[2].legend(loc='lower right', frameon=True)
# plt.xscale('log')


# fig, ax = plt.subplots()
# plt.figure(2)
# plt.plot(time1, T1p, color='red', marker='o', linestyle='', markersize=0.7, label=r'$T_{tripped}$')
# ax.plot(time2, T2p, 'bo', markersize=0.7, label=r'$T_{non-tripped}$')
# ax.set(xlabel='Time (s)', ylabel='Temperature (K)')
# ax.grid(True, linestyle='--', linewidth=0.3)
# ax.legend(loc='upper left', frameon=True)
# # plt.xlim(-70,2600)
# plt.ylim(-1, 13)

# plt.title('test')

# plt.plot(pointX, pointY, 'go', markersize=5)
# plt.plot(lineX, lineY, 'r--', linewidth=0.8)
# plt.errorbar(X, Y, yerr=0.1, fmt='ko', linewidth=0.8, capsize=3, capthick=0.8, markersize=5)
# plt.xlabel(r'$\textbf{Time } (ms)$')
# plt.ylabel(r'$\textbf{Temperature } (K)$')
# i="front"
# plt.savefig("Graphen/Temperature_" + str(i) + ".png")

plt.show()
