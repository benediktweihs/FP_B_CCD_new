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



#alle Graphen werden in Graphen gespeichert
if not os.path.exists("Graphen"):
    os.mkdir("Graphen")

data = convert("test")
########################################
#alle werte in cm bzw cm/s
#######################################
Data1 = np.array(data[11], dtype=float)
Data2 = np.array(data[12], dtype=float)
Data3 = np.array(data[13], dtype=float)
Data4 = np.array(data[14], dtype=float)


v1=[]   #v1 vor stoß DATA2
v2=[]   #v2 vor stoß DATA3
v1prime=[]   #v1 nach stoß
v2prime=[]   #v2 nach stoß
vrel=[]
vrelprime=[]
vsp=[]
vspprime=[]
DELTAvsp=[]
krel=[]
krelprime=[]
eta=[]
ms=ufloat(180,2)
mz=ufloat(10,0.1)
mg=ufloat(50,0.5)
m2=ms+mz
m1=ms+mz

#messunsicherheit geschwindigkeit:
#+-0.1cm/s
# 3*10^-5 * v/(cm/s)
# fehler für v berechnen
def FehlerV(v):
    return np.sqrt(0.01+(3*10**(-5)*v*v)**2)

for i in range(len(Data1)):
    v1.append(ufloat(Data1[i], FehlerV(Data1[i])))
    v2.append(ufloat(Data3[i], FehlerV(Data3[i])))
    v2prime.append(ufloat(Data4[i], FehlerV(Data4[i])))
    v1prime.append(ufloat(Data2[i], FehlerV(Data2[i])))

#Berechnung von vsp und vrel
for i in range(len(Data1)):
    vrel.append(v1[i]-v2[i])
    vsp.append((m1*v1[i]+m2*v2[i])/(m1+m2))
    vrelprime.append(v1prime[i] - v2prime[i])
    vspprime.append(v2prime[i])
    DELTAvsp.append(vspprime[i]-vsp[i])


#print(vsp)
#print(vspprime)

#Berechnung von Krel und Krel'
mü=(m1*m2)/(m1+m2)
f=[]
for i in range(len(Data1)):
    krel.append(0.5*mü*(vrel[i]/100)**2)
    krelprime.append(0.5 * mü * (vrelprime[i]/100) ** 2)
    eta.append((vrelprime[i]/vrel[i])**2)
    #f.append(-krelprime[i]+krel[i])
    print('{:.1u}'.format(eta[i]))

#print(krel)
#print(krelprime)
#print(f)
#print(eta)
#print(v1)
#print(v1prime)

#print(DELTAvsp)
#G=0
#for i in range(len(Data1)):
    #G+=g[i]
    #print(d[i])
    #print("\t")
    #print(v1[i])
    #print("\t")
    #print(v2[i])
    #print("\n")

#G=G/5
#print('G mittel =')
#print(G)
#Gsys = 0.02*G
#print('systematischer Fehler von G ist:')
#print(Gsys)
