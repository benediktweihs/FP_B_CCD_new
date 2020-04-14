from __future__ import unicode_literals
from uncertainties import *
from converterNew import *
from uncertainties import unumpy
from uncertainties import *
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



#alle Graphen werden in Graphen gespeichert
if not os.path.exists("Graphen"):
    os.mkdir("Graphen")

#data enthält sämtliche Information
data = np.array(convert("messwerteText"))

sumN = 0
faktorN = 0
sumAr=0
faktorAr = 0
message=''
for i in range(7):
    if i>=4:
        message = ''
    X = data[2*i]
    X = [a for a in X if a != 'a']
    X = np.array(X, dtype=float)
    Y = data[2*i+1]
    Y = [a for a in Y if a != 'a']
    Y = np.array(Y, dtype=float)

    deltap = Y[0]
    Y = np.delete(Y,[0])
    deltaT = X[1]
    X = np.delete(X,[0,1])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(i+1)

    k = (Y[1]-Y[0])/(X[1]-X[0])
    lineX = np.linspace(-deltaT-2,20,1000)
    lineY = k*(lineX-X[0])+Y[0]
    pointX = [-deltaT]
    pointY = [k*(-deltaT-X[0])+Y[0]]
    #deltapPrime = Y[len(Y)-1]-pointY[0]
    #Kappa = deltap/(deltap-deltapPrime)

    #Abbidlungen
    plt.annotate(message, xy=(X[len(X)-1]-2.5,k*((-deltaT-2)-X[0])+Y[0]), fontsize=18)
    plt.plot(pointX,pointY, 'go', markersize=5)
    plt.plot(lineX, lineY, 'r--', linewidth=0.8)
    plt.errorbar(X, Y, yerr = 0.1, fmt='ko',linewidth=0.8, capsize=3,capthick=0.8,markersize=5)
    plt.xlabel(r'$\textbf{time } (s)$')
    plt.ylabel(r'$\textbf{Druck } (cmH_2O)$')
    plt.savefig("Graphen/adiabatisch"+str(i)+".pdf")

    #Gerade um deltaT nach hinten erweitern
    FehlerZeit = 0.1 #nein!
    y0 = ufloat(Y[0], 0.1)
    y1 = ufloat(Y[1], 0,1)
    x0 = ufloat(X[0], FehlerZeit)
    x1 = ufloat(X[1], FehlerZeit)
    k = (y1-y0)/(x1-x0)
    deltaT = ufloat(deltaT, FehlerZeit)
    pointy = k * (-deltaT - x0) + y0

    #deltapPrime ausrechnen und damit Kappa
    ymax = ufloat(Y[len(Y)-1], 0.1)
    deltapprime=ymax-pointy
    deltap = ufloat(deltap, 0.1)
    Kappa = deltap / (deltap - deltapprime)
    print(Kappa)
    #gewichteter Mittelwert mit Fehler
    if i < 4:
        sumN = sumN + (1/(std_dev(Kappa)**2))*nominal_value(Kappa)
        faktorN = faktorN + (1/(std_dev(Kappa)**2))
    else:
        sumAr = sumAr + (1/(std_dev(Kappa)**2))*nominal_value(Kappa)
        faktorAr = faktorAr + (1/(std_dev(Kappa)**2))

#print("gemitteltes Kappa Luft = " + str(sumN/faktorN))
#print("Std.-Abweichung Luft = " + str(np.sqrt(1/faktorN)))
#print("gemitteltes Kappa Argon = " + str(sumAr/faktorAr))
#print("Std.-Abweichung Argon = " + str(np.sqrt(1/faktorAr)))