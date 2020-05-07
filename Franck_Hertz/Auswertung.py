import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from Franck_Hertz.converter import convert
from uncertainties import *
from uncertainties import unumpy
from matplotlib import rc
from scipy.special import voigt_profile

rc('text', usetex=True)

parent = os.path.dirname(os.path.dirname(__file__))
t = ufloat(9.14, 0.01)
errY = 0.03  # 2 prozent

# unwichtig... arrays schneiden usw.
def parabel(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def gaussian(x, a, b, c, d):
    return a*(np.e**(-((x-c)**2)/b)) + d
    #return a*voigt_profile(x-c, b, d) + e

def linear(x, a, d):
    return a*x + d

def minimum(a, b):
    return -b/(2*a)

def plotFile(file):
    data = np.array(convert(file, ',', 16), dtype=float)
    x = data[1]
    y = data[3]
    plt.plot(x, y, 'bo', markersize=1)
    plt.show()
    plt.close()

def cutMin(arr, compare, minimum):
    temp = []
    for i, x in enumerate(arr):
        if compare[i]>minimum:
            temp.append(x)
    return np.array(temp)

def minMitFehlerAusFit(popt, pcov):
    a = ufloat(popt[0], np.sqrt(pcov[0][0]))
    b = ufloat(popt[1], np.sqrt(pcov[1][1]))
    return minimum(a, b)

def cut(arr, compare, minimum, maximum):
    temp = []
    for i, x in enumerate(arr):
        if compare[i]<maximum and compare[i]>minimum:
            temp.append(x)
    return np.array(temp)

def parabolaFit(file, minimum, boundary, show):
    if show:
        plt.xlabel(r'$\frac{U_B}{t}$ $(V)$')
        plt.ylabel(r'$U_K \propto I_K$ $(V)$')
    name = ""
    for f in file:
        if f=='.': break
        name = name + f
    data = np.array(convert(file, ',', 16), dtype=float)
    x = cutMin(data[1], data[3], minimum)
    y = cutMin(data[3], data[3], minimum)
    if show: plt.plot(x, y, 'bo', markersize=1)

    minima = []
    # fitte parabel zu minima
    for i, x0 in enumerate(boundary):
        xLower, xUpper = boundary[i][0], boundary[i][1]
        domainFit = cut(x, x, xLower, xUpper)
        coDomainFit = cut(y, x, xLower, xUpper)
        popt, pcov = curve_fit(parabel, domainFit, coDomainFit)
        xFit = np.linspace(domainFit[0], domainFit[-1], 1000)
        yFit = parabel(xFit, *popt)
        if show: plt.plot(xFit, yFit, 'r-', lw=3)
        print(minMitFehlerAusFit(popt, pcov))
        minima.append(minMitFehlerAusFit(popt, pcov))

    if show:
        plt.savefig("C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\allesUnkor" + name + ".pdf")
        plt.savefig("C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\allesUnkor" + name + ".png")
        plt.close()
    return minima

def offset(n):
    retArrOnes, retArrTwos = [], []
    N = len(n)
    ones, twos, help = [], [], []
    for i in n:
        if i == 1:
            ones.append(1)
            help.append(1)
        else:
            twos.append(2)
            help.append(2)
    for temp in [ones, twos]:
        if temp[0]==1: retArr = retArrOnes
        else: retArr = retArrTwos
        for counter, i in enumerate(temp):
            if N % 2 == 1 and counter==0:
                retArr.append(i)
                flag=True

            elif N % 2 == 0 and counter==0:
                retArr.append(i + ((-1) ** counter) * 0.01 * ((counter + 1) - counter % 2))
                flag=False

            else:
                if flag: counter -= 1
                retArr.append(i+((-1)**counter)*0.01*((counter+1)-counter%2))
    tot = []
    c1, c2 = 0, 0
    for c, i in enumerate(n):
        if help[c]==1:
            tot.append(retArrOnes[c1])
            c1+=1
        else:
            tot.append(retArrTwos[c2])
            c2+=1
    return tot


def gaussianFit(file, minimum, boundary, boundaryMax, show, cutUpperOfPlot):
    name = ""
    for f in filename:
        if f=='.': break
        name = name + f
    data = np.array(convert(file, ',', 16), dtype=float)
    x = cutMin(data[1], data[3], minimum)
    y = cutMin(data[3], data[3], minimum)
    #if show: plt.figure(figsize=(5, 4))
    if show:
        plt.xlabel(r'$\frac{U_B}{t}$ $(V)$')
        plt.ylabel(r'$U_K \propto I_K$ $(V)$')
        plt.plot(x, y, 'bo', markersize=1)

    maxima, maximaY = [], []
    # fitte Maxima und ziehe Parable ab aka einhüllende bestimmen
    for i, x0 in enumerate(boundaryMax):
        xLower, xUpper = x0[0], x0[1]
        domainFit = cut(x, x, xLower, xUpper)
        coDomainFit = cut(y, x, xLower, xUpper)  # TODO error y -> 2% of value
        popt, pcov = curve_fit(parabel, domainFit, coDomainFit, sigma=errY*coDomainFit)  # WENN DOCH GAUSS HIER ÄNDERN
        #print(filename + str(popt))
        redChiSqu = sum(((coDomainFit - parabel(domainFit, *popt)) / (errY*coDomainFit)) ** 2) / (len(domainFit) - 3)
        fitPar = unumpy.uarray(popt, np.sqrt(np.diag(pcov)))
        xFit = np.linspace(domainFit[0], domainFit[-1], 1000)
        yFit = parabel(xFit, *popt)  # WENN DOCH GAUSS HIER ÄNDERN
        maximum = minMitFehlerAusFit(popt, pcov)
        maximumY = parabel(maximum, *fitPar) # WENN DOCH GAUSS HIER ÄNDERN
        if show: plt.plot(xFit, yFit, 'y-', lw=2)
        maxima.append(maximum)
        maximaY.append(maximumY)
    if show: plt.plot(unumpy.nominal_values(maxima), unumpy.nominal_values(maximaY), 'kx', markersize=3)

    # einhüllende fitten und abziehen
    poptTot, pcovTot = curve_fit(parabel, unumpy.nominal_values(maxima), unumpy.nominal_values(maximaY), sigma=unumpy.std_devs(maximaY))
    xFitTot = np.linspace(x[0], x[-1], 1000)
    if show:
        plt.plot(xFitTot, parabel(xFitTot, *poptTot), 'r-', lw=2)
        plt.savefig("C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\maxFit" + name + ".pdf")
        plt.savefig("C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\maxFit" + name + ".png", dpi=400)
        plt.close()

    poptLow = [p + np.sqrt(pcov[i][i])/2 for i, p in enumerate(poptTot)]
    poptHigh = [p - np.sqrt(pcov[i][i])/2 for i, p in enumerate(poptTot)]
    minMean = gaussHelp(x, y, boundary, poptTot, show, name, cutUpperOfPlot)
    minLow = gaussHelp(x, y, boundary, poptLow, True, name +"low", cutUpperOfPlot)
    minHigh = gaussHelp(x, y, boundary, poptHigh, True, name +"high", cutUpperOfPlot)
    mReturn = []
    for i, m in enumerate(minMean):
        errExtra = max(np.abs(nominal_value(m) - nominal_value(minLow[i])), np.abs(nominal_value(m) - nominal_value(minHigh[i])))
        mReturn.append(ufloat(nominal_value(m), np.sqrt(std_dev(m)**2 + errExtra**2)))
    return mReturn

def gaussHelp(x, y, boundary, poptTot, show, file, cutUpperOfPlot):
    #if show: plt.figure(figsize=(5, 3))
    if show:
        plt.xlabel(r'$\frac{U_B}{t}$ $(V)$')
        plt.ylabel(r'$U_K - U_{einh.} \propto I_K - I_{einh.}$ $(V)$')
    print(cutUpperOfPlot)
    minima = []
    y=parabel(x, *poptTot)-y
    yeff, xeff = [], []
    for i, elem in enumerate(y):
        if elem < cutUpperOfPlot:
            yeff.append(elem)
            xeff.append(x[i])
    if show: plt.plot(xeff, yeff, 'bo', markersize=1)

    # fitte Gaussfunktion zu minima/maxima
    for i, x0 in enumerate(boundary):
        xLower, xUpper = boundary[i][0], boundary[i][1]
        domainFit = cut(x, x, xLower, xUpper)
        coDomainFit = cut(y, x, xLower, xUpper)
        popt, pcov = curve_fit(gaussian, domainFit, coDomainFit, p0 = [.1, 1, xLower + (xUpper-xLower)/2, 0])  #GAUSS
        xFit = np.linspace(domainFit[0], domainFit[-1], 1000)
        yFit = gaussian(xFit, *popt)
        minimum = ufloat(popt[2], np.sqrt(pcov[2][2]))  # GAUSS
        #minimum = minMitFehlerAusFit(popt, pcov)  #PARABEL
        #plt.plot(xFit, gaussian(xFit, .1, 1, xLower + (xUpper-xLower)/2, 0))
        if show:
            plt.plot(xFit, yFit, 'r-', lw=2)
        minima.append(minimum)
    if show:
        plt.savefig(
            "C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\korrigiert" + file + ".pdf", bbox_inches='tight')
        plt.savefig(
            "C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\korrigiert" + file + ".png", dpi=400, bbox_inches='tight')
        plt.close()

    if show:
        plt.show()
        plt.close()
    return minima

def determineEa(minimaTot, name):
    y, n = [], []
    plt.ylabel(r'$\frac{\Delta U_B(n)}{t}$ $(V)$')
    plt.xlabel(r'$n$')
    for i, arr in enumerate(minimaTot):
        y.append(arr[1]-arr[0])
        n.append(1)
        if len(arr) == 3:
            y.append(arr[2]-arr[1])
            n.append(2)
    popt, pcov = curve_fit(linear, n, unumpy.nominal_values(y), sigma=unumpy.std_devs(y))
    nFit = np.linspace(.5,2,1000)
    faktor = nominal_value(t)
    yFit = linear(nFit, faktor*popt[0], faktor*popt[1])
    plt.errorbar(offset(n), faktor*unumpy.nominal_values(y), yerr=faktor*unumpy.std_devs(y), fmt='ro', linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
    plt.plot(offset(n), faktor*unumpy.nominal_values(y), 'ko', markersize=2)
    plt.plot(nFit, yFit, 'k-', lw=.6)
    plt.savefig(
            "C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Graphen\\final" + name + ".pdf")

    plt.close()
    a, b = ufloat(popt[0], np.sqrt(pcov[0][0])), ufloat(popt[1], np.sqrt(pcov[1][1]))
    chisqu = sum(((linear(np.array(n), *popt) - np.array(unumpy.nominal_values(y)))/np.array(unumpy.std_devs(y)))**2)/(len(n)-2)
    print("a = " + str(a))
    return linear(.5, a, b), chisqu, a

if __name__ == '__main__':
    # problem bei t0002ALL (für gaussfit) is dass ma maximal zwei maxima fitten kann... zwei punkte mit einer geraden zu fitten is so a sache.
    directory = "C:\\Users\\Benedikt Weihs\\PycharmProjects\\FP_B_CCD_new\\Franck_Hertz\\Data\\"  # working directory
    estimations = [
        [[2.5, 3.3], [4.5, 5.35], [6.4, 7.2]],
        [[2.4, 3.35], [4.35, 5.25]],
        [[2.65, 3.5], [4.6, 5.6]],
        [[2.5, 3.3], [4.5, 5.4], [6.3, 7.2]],
        [[2.5, 3.35], [4.55, 5.4], [6.4, 7.3]],
        [[2.55, 3.5], [4.55, 5.6]],
        [[2.5, 3.3], [4.4, 5.4], [6.4, 7.2]] # unused
    ]
    filesParabola = ["T0001ALL.CSV", "T0004.CSV"]  # abbildungen die nit in sättigun sin TODO t0005.csv?
    estimationsMax = [
        [[1.6, 2.8], [3.5, 4.6], [5.6, 6.8]],
        [[1.6, 2.8], [3.5, 4.7], [5.7, 6.8]]
    ]  # maxima bei abbildungen die nit in sättigung sin
    estimationsMin = [
        [[2.2, 3.9], [4.2, 5.85], [6.2, 7.7]],
        [[2.35, 3.75], [4.3, 5.8], [6.3, 7.8]]
    ]  # gleiche zeilen wie in estimations nur halt für die Dateien in filesParabola
    minimaTot = []

    # mit parabel fit:
    # E_a und lambda berechnen:
    # TODO auswahlfehler aka Auswirkung von boundary aufs Ergebnis
    for i, filename in enumerate(os.listdir(directory)):
        # FIT DER MINIMA MIT PARABEL
        print(filename)
        if filename == "T0007.CSV": continue
        minimaTot.append(parabolaFit(filename, 0.03, estimations[i], True))  # nahe null auf der y achse ganz viele datenpunkte
    u, chisqu, a = determineEa(minimaTot, "stupid")
    print("stupid way: ")
    print("___Lambda___ " + str((5e-3/(2*t*u))*(a*t)))
    print("\n___E_a___ = " + str(t*u))
    print("\n___red_chisqu___ = " + str(chisqu))

    minimaTot = []
    mx = [.13, .4]
    # Korrektur durch abziehen der Kennlinie ohne Minima - geht nur wenn Signal nicht in sättigung ist.
    for i, filename in enumerate(filesParabola):
        # FIT DER MAXIMA MIT GAUSS + FIT DER EINHÜLLENDEN MIT PARABEL + ABZIEHEN DER PARABEL + FITTE MINIMA MIT GAUSS
        minimaTot.append(gaussianFit(filename, 0.03, estimationsMin[i], estimationsMax[i], True, mx[i]))
        #print(mx[i])
    u, chisqu, a = determineEa(minimaTot, "korrigiert")
    print("___Lambda___ " + str((5e-3/(2*t*u))*(a*t)))
    print("\n___E_a___ = " + str(t*u))
    print("\n___red_chisqu___ = " + str(chisqu))

    # zeigt einfach nur daten an
    #for i, filename in enumerate(os.listdir(directory)):
    #    plotFile(filename)
    #    print(filename)

