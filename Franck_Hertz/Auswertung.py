import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from Franck_Hertz.converter import convert
from uncertainties import *
from uncertainties import unumpy

# unwichtig... arrays schneiden usw.
def parabel(x, a, b, c):
    return a*x**2 + b*x + c
def gaussian(x, a, b, c, d):
    return a*np.exp(-b*(x-c)**2) + d
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
        plt.show()
        plt.close()
    return minima
def offset(i):
    if i%2 == 0:
        return 0.02*((i//2)+1)
    if i%2 == 1:
        return 0.02*((i//2)-1)
def gaussianFit(file, minimum, boundary, boundaryMax, show):
    data = np.array(convert(file, ',', 16), dtype=float)
    x = cutMin(data[1], data[3], minimum)
    y = cutMin(data[3], data[3], minimum)
    if show: plt.plot(x, y, 'bo', markersize=1)

    minima, maxima, maximaY = [], [], []

    # fitte Maxima und ziehe Parable ab
    for i, x0 in enumerate(boundary):
        xLower, xUpper = x0[0], x0[1]
        domainFit = cut(x, x, xLower, xUpper)
        coDomainFit = cut(y, x, xLower, xUpper)
        popt, pcov = curve_fit(gaussian, domainFit, coDomainFit)  # WENN DOCH PARABEL HIER ÄNDERN
        xFit = np.linspace(domainFit[0], domainFit[-1], 1000)
        yFit = gaussian(xFit, *popt)  # WENN DOCH PARABEL HIER ÄNDERN
        maximum = ufloat(popt[2], np.sqrt(pcov[2][2]))
        maximumY = gaussian(nominal_value(maximum), *[ufloat(popt[i], np.sqrt(pcov[i][i])) for i in range(4)])  # WENN DOCH PARABEL HIER ÄNDERN
        if show: plt.plot(xFit, yFit, 'b-', lw=3)
        maxima.append(maximum)
        maximaY.append(maximumY)

    if show: plt.errorbar(unumpy.nominal_values(maxima), unumpy.nominal_values(maximaY), yerr=unumpy.std_devs(maximaY), fmt='ro', linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
    poptTot, pcovTot = curve_fit(parabel, unumpy.nominal_values(maxima), unumpy.nominal_values(maximaY), sigma=unumpy.std_devs(maximaY))
    y=y-parabel(x, *poptTot)

    # fitte Gaussfunktion zu minima
    for i, x0 in enumerate(boundary):
        xLower, xUpper = boundary[i][0], boundary[i][1]
        domainFit = cut(x, x, xLower, xUpper)
        coDomainFit = cut(y, x, xLower, xUpper)
        popt, pcov = curve_fit(gaussian, domainFit, coDomainFit)
        xFit = np.linspace(domainFit[0], domainFit[-1], 1000)
        yFit = parabel(xFit, *popt)
        minimum = ufloat(popt[2], np.sqrt(pcov[2][2]))
        if show: plt.plot(xFit, yFit, 'r-', lw=3)
        print("Minimum mit Gauß fit: " + str(minimum))
        minima.append(minimum)

    if show:
        plt.show()
        plt.close()
    return minima
def determineEa(minimaTot):
    y, n = [], []
    for i, arr in enumerate(minimaTot):
        y.append(arr[1]-arr[0])
        n.append(1)
        if len(arr) == 3:
            y.append(arr[2]-arr[1])
            n.append(2)
    popt, pcov = curve_fit(linear, n, unumpy.nominal_values(y), sigma=unumpy.std_devs(y))
    nFit = np.linspace(.5,2,1000)
    yFit = linear(nFit, *popt)
    plt.errorbar(n, unumpy.nominal_values(y), yerr=unumpy.std_devs(y), fmt='ro', linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
    plt.plot(n, unumpy.nominal_values(y), 'ko', markersize=2)
    plt.plot(nFit, yFit, 'k-', lw=.6)
    plt.show()
    plt.close()
    a, b = ufloat(popt[0], np.sqrt(pcov[0][0])), ufloat(popt[1], np.sqrt(pcov[1][1]))
    chisqu = sum(((linear(np.array(n), *popt) - np.array(unumpy.nominal_values(y)))/np.array(unumpy.std_devs(y)))**2)/(len(n)-2)
    print("a = " + str(a))
    return linear(.5, a, b), chisqu, a

if __name__ == '__main__':
    directory = "/home/benedikt/PycharmProjects/FP_B_CCD_Patrick/Franck_Hertz/Data"
    estimations = [
        [[2.5, 3.3], [4.5, 5.55]],
        [[2.55, 3.35], [4.45, 5.45], [6.4, 7.2]],
        [[2.5, 3.3], [4.4, 5.4], [6.4, 7.2]], # unused
        [[2.5, 3.3], [4.4, 5.4], [6.4, 7.2]],
        [[2.4, 3.3], [4.35, 5.25]],
        [[2.65, 3.5], [4.55, 5.65]],
        [[2.5, 3.3], [4.4, 5.4], [6.4, 7.2]]
    ]
    # TODO estimationsMax und files in array bei der Einhüllende fitten Sinn macht.
    # TODO auswahlfehler aka Auswirkung von boundary aufs Ergebnis
    minimaTot = []
    for i, filename in enumerate(os.listdir(directory)):
        print(filename)
        if filename == "T0007.CSV": continue
        minimaTot.append(parabolaFit(filename, 0.03, estimations[i], True))  # nahe null auf der y achse ganz viele datenpunkte
    u, chisqu, a = determineEa(minimaTot)


    t = ufloat(9.14, 0.01)
    #print("lambda " + str(((1.602*1e-19*5e-3)/(2*t*u))*(a/t)))
    print("\n___E_a___ = " + str(t*u))
    print("\n___red_chisqu___ = " + str(chisqu))

    # zeigt einfach nur daten an
    #for i, filename in enumerate(os.listdir(directory)):
    #    plotFile(filename)
    #    print(filename)

