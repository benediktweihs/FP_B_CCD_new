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

#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

#alle Graphen werden in Graphen gespeichert
if not os.path.exists("Graphen"):
    os.mkdir("Graphen")

#data enthält sämtliche Information
#data = np.array(convert("messwerteText"))

#data = convert("test")
data = convert("XY_dämpfung")
#print(data[0])
x = np.array(data[0], dtype=float)
#x = x[0::100]
y = np.array(data[1], dtype=float)
#y = y[0::100]
plt.plot(x,y,'kx',markersize = 1)
plt.show()

########################################
#alle werte in cm bzw cm/s
#######################################
Data1 = np.array(data[0], dtype=float)
Data2 = np.array(data[1], dtype=float)
Data3 = np.array(data[2], dtype=float)
d=[] #Abstand DATA1
deltaD=0.2 #kann auch kleiner gewählt werden
h=1.55 # höhe der neigung
deltaH=0.02
h=ufloat(h,deltaH)
s=100 # länge für neigung
deltaS=0.5
s=ufloat(s,deltaS)

v1=[]   #v1 vor stoß DATA2
v2=[]   #v2 vor stoß DATA3
v1prime=[]   #v1 nach stoß
v2prime=[]   #v2 nach stoß
g=[]

#messunsicherheit geschwindigkeit:
#+-0.1cm/s
# 3*10^-5 * v/(cm/s)
# fehler für v berechnen
def FehlerV(v):
    return np.sqrt(0.01+(3*10**(-5)*v*v)**2)


for i in range(len(Data1)):
    d.append(ufloat(Data1[i],deltaD))
    v1.append(ufloat(Data2[i], FehlerV(Data2[i])))
    v2.append(ufloat(Data3[i], FehlerV(Data3[i])))
    #v2prime.append(ufloat(Data1[i], FehlerV(Data1[i])))
    #v1prime.append(ufloat(Data1[i], FehlerV(Data1[i])))

#Berechnung des Neigungswinkels
alpha = unumpy.arctan(h/s)
#Berechnung von g
aneu=0
gneu=[]
G=0
A=0
for i in range(len(Data1)-1):
    a=(v2[i]**2-v1[i]**2)/(2*d[i])
    g.append(a/unumpy.sin(alpha)/100)
    aneu = ((0.99*v2[i]) ** 2 - (0.99*v1[i]) ** 2) / (2 * d[i])
    gneu.append(aneu / unumpy.sin(alpha) / 100)
    #print('{:.1u}'.format(a))
    #print('{:.1u}'.format(aneu))
    A += aneu
    print('{:.1u}'.format(d[i]))



#print(g)

for i in range(len(Data1)-1):
    #A+=a[i]
    G+=gneu[i]
    #print('{:.1u}'.format(gneu[i]))
    #print("\t")
    #print(v1[i])
    #print("\t")
    #print(v2[i])
    #print("\n")

G=G/5
A=A/5
#print('A mittel =')
#print('{:.1u}'.format(A))
