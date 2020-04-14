import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from uncertainties import *
from scipy.optimize import curve_fit
from converterNew import *
from matplotlib import rc
parent = os.path.dirname(os.path.dirname(__file__))  # direcotry
rc('text', usetex=True)  # Latex


# table.txt ist Ausgabe von aufgabe_4.prg und aufgabe_4_b.prg
data = np.array(convert("table", "txt", ';'), dtype=float)
data_b = np.array(convert("table_b", "txt", ';'), dtype=float)
mean, std = data[0], data[1]
meanB, stdB = data_b[0], data_b[1]
sigma_ron = 7.14
#delta_sigma_ron = i need U Patrick

# n_adu(sigma_tot) mit Parabel fkt. fitten
def fit(x, a, b, c):
    return a*x**2 + b*x + c
def fit2(x, a, b):
    return a*x**2 + b*x + sigma_ron**2
def fitCheckPlus(x, a, b):
    return a * x ** 2 + b * x + (sigma_ron+delta_sigma_ron) ** 2
def fitCheckMinus(x, a, b):
    return a * x ** 2 + b * x + (sigma_ron-delta_sigma_ron) ** 2
cutMax, cutMaxB = 4, 5
popt, pcov = curve_fit(fit, mean[0:len(mean)-cutMax:], std[0:len(mean)-cutMax:]**2)
poptB, pcovB = curve_fit(fit, meanB[0:len(meanB)-cutMaxB:], stdB[0:len(meanB)-cutMaxB:]**2)
popt2, pcov2 = curve_fit(fit2, mean[0:len(mean)-cutMax:], std[0:len(mean)-cutMax:]**2)
poptB2, pcovB2 = curve_fit(fit2, meanB[0:len(meanB)-cutMaxB:], stdB[0:len(meanB)-cutMaxB:]**2)
xFit = np.linspace(0, mean[-1], 1000)
yFit = fit(xFit, *popt)
yFitB = fit(xFit, *poptB)
yFit2 = fit2(xFit, *popt2)
yFitB2 = fit2(xFit, *poptB2)
print("_____FIT_PAR_GRÜN_N-3DOF_____    "+str(popt))
print("_____FIT_PAR_BLAU_N-3DOF_____    "+str(poptB))
print("_____FIT_PAR_GRÜN_N-2DOF_____    "+str(popt2))
print("_____FIT_PAR_BLAU_N-2DOF_____    "+str(poptB2))


# alles plotten
scaleX, scaleY = 1e-4, 1e-4
plt.plot(scaleX*mean[len(mean)-cutMax:len(mean)-1:], scaleY*std[len(mean)-cutMax:len(mean)-1:]**2, 'go', markersize=5)
plt.plot(scaleX*meanB[len(meanB)-cutMaxB:len(meanB)-1:], scaleY*stdB[len(meanB)-cutMaxB:len(meanB)-1:]**2, 'bo', markersize=5)
plt.plot(scaleX*meanB[-1], scaleY*stdB[-1]**2, 'o', color="purple", markersize=5)  # haha... sie liegen einfach perfekt aufeinander

plt.plot(scaleX*meanB[0:len(meanB)-cutMaxB:], scaleY*std[0:len(meanB)-cutMaxB:]**2, 'bx', markersize=5)
plt.plot(scaleX*xFit, scaleY*yFitB, 'b-', lw=2)
#plt.plot(scaleX*xFit, scaleY*yFitB2, 'b--', lw=2)  # fit mit 2/3 fitparameter ist gleich

plt.plot(scaleX*mean[0:len(mean)-cutMax:], scaleY*std[0:len(mean)-cutMax:]**2, 'gx', markersize=5)
plt.plot(scaleX*xFit, scaleY*yFit, 'g-', lw=2)
#plt.plot(scaleX*xFit, scaleY*yFit2, 'g--', lw=2)  # fit mit 2/3 fitparameter ist gleich

plt.savefig(parent + "/PlottingGp-master/Graphen/aufgabe5.pdf")


# gain faktor mehr oder weniger schon berechnet:
popt2CheckM, pcov2CheckM = curve_fit(fitCheckMinus, mean[0:len(mean)-cutMax:], std[0:len(mean)-cutMax:]**2)
poptB2CheckM, pcovB2CheckM = curve_fit(fitCheckMinus, mean[0:len(mean)-cutMax:], std[0:len(mean)-cutMax:]**2)
popt2CheckP, pcov2CheckP = curve_fit(fitCheckPlus, meanB[0:len(meanB)-cutMaxB:], stdB[0:len(meanB)-cutMaxB:]**2)
poptB2CheckP, pcovB2CheckP = curve_fit(fitCheckPlus, meanB[0:len(meanB)-cutMaxB:], stdB[0:len(meanB)-cutMaxB:]**2)

# sigma_ron hat ja auch einen Fehler... und taucht aber in fitfunktion auf
# standard Vorgehensweise is jetzt den fit dreimal zu machen.. einmal mit mittwert für sigma_ron
# und zweimal mit extremwerten --> soviel wie dann da fitparamtere geändert wird is dann a neuer Fehler
# dann nur mehr Fehler aus ursprünglichem fit mit neuem quadratisch addieren und fertig
deltaG = max(popt2[1] - popt2CheckM[1], popt2[1] - popt2CheckP[1])
deltaB = max(popt2[1] - poptB2CheckM[1], popt2[1] - poptB2CheckP[1])
errG2 = np.sqrt(pcov2[1][1] + deltaG**2)
errB2 = np.sqrt(pcovB2[1][1] + deltaB**2)
gain_seperat = unumpy.uarray([popt2[1], poptB2[1]], [errG2, errB2])
gain_seperat2 = unumpy.uarray([popt[1], poptB[1]], [pcov[1][1]**.5, pcovB[1][1]**.5])  # nebensächlich
print("GAIN GRÜN-BLAU MIT FIXEM OFFSET: " + str(1/gain_seperat))
print("GAIN GRÜN-BLAU MIT OFFSET ALS FITPAR: " + str(1/gain_seperat2))
print("GAIN FIX OFFSET: " + str(meanValue(gain_seperat**(-1))[0]) + "+/-" + str(meanValue(gain_seperat**(-1))[1]))
print("GAIN VAR OFFSET: (UNWICHTIG) " + str(meanValue(gain_seperat2**(-1))[0]) + "+/-" + str(meanValue(gain_seperat2**(-1))[1]))