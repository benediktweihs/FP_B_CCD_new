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

#data = convert("Temperatur")
data = convert("UI_front")
time = np.array(data[0], dtype=float)
Vbat = np.array(data[2], dtype=float)
V2 = np.array(data[3], dtype=float)

fit =np.array(3.96125470e+00+time*(-2.87912582e-07))
fit2 =np.array(3.94385322+time*(-2.69122910e-07))
fit3 =np.array(3.95094740e+00+time*(-2.78023652e-07))   ##gewählt
fit4 =np.array(3.96195974e+00+time*(-2.87507258e-07))


verteilung=np.array(Vbat-fit)
verteilung2=np.array(Vbat-fit2)
verteilung3=np.array(Vbat-fit3)
verteilung4=np.array(Vbat-fit4)

#print(fit)

#vbat1=Vbat[0:1000:]
vbat2=Vbat[1000:8000:]
t2=time[1000:8000:]
vbat3=Vbat[2000:6000:]
t3=time[2000:6000:]
vbat4=Vbat[3900:4600:]
t4=time[3900:4600:]

vert=verteilung
vert2=verteilung2[1000:8000:]
vert3=verteilung3[2000:6000:]
vert4=verteilung4[3900:4600:]

linfit=np.polyfit(time,Vbat,1)
print(linfit)
linfit=np.polyfit(t2,vbat2,1)
print(linfit)
linfit=np.polyfit(t3,vbat3,1)
print(linfit)
linfit=np.polyfit(t4,vbat4,1)
print(linfit)


#vbat4=Vbat[855737:856000:]
#plt.hist(vbat1,bins=100)
#plt.hist(vbat2, bins=500)
plt.figure(1)
plt.hist(verteilung3*1000, bins=500)
plt.hist(vert3*1000, bins=500)
#print(vbat4)
#plt.figure(2)
#plt.hist(vert, bins=500)
#print(np.histogram(verteilung,bins=10))
#print(np.std(vert2))
#print(np.std(vert3))
print(np.std(vert3))
print(np.std(verteilung3))
plt.show()

#print(data[0])
#time1 = np.array(data[0],dtype=float)
#time1 = np.concatenate((time1[0:100:10],time1[100::100]))

#print(time1)
#t1_1=time1[0:100:2]
#t1_2=time1[101::10]
#t1=np.hstack((t1_1,t1_2))
#t1=np.hstack((time1[0:100:2],time1[101::10]))
##print(t1)
#print(time1)
#time1=time1[0::2]
#time2 = np.array(data[3],dtype=float)
#time2=time2[0::2]
#T1a = np.array(data[2], dtype=float)
#T1a=T1a[0::2]
#T1p = np.array(data[1], dtype=float)
#T1p=T1p[0::2]
#T2a = np.array(data[5], dtype=float)
#T2a=T2a[0::2]
#T2p = np.array(data[4], dtype=float)