import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from Emissionsspektroskopie.converter import *
from numpy import *
from uncertainties import unumpy
from uncertainties import *
import scipy.constants as con


directory = "C:\\Users\\weihs\\PycharmProjects\\FP_B_CCD_new\\Emissionsspektroskopie\\Data2\\"
background = np.array(convert("a.txt", ';', 78, "Data2"), dtype=float)
names = ["b", "d", "f", "h"]
formatStrings = ["r", "y", "g", "b"]
filenames = [n+".txt" for n in names]


# kennlinien aus daten:
I_gruen = [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.50, 2.00, 3.30, 4.10, 5.00]
V_gruen = [0.000, 0.500, 1.000, 1.500, 2.00, 2.43, 2.49, 2.54, 2.60, 2.67, 2.75, 2.80, 2.84, 2.89, 2.99, 3.03, 3.07]
I_blau = [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.50, 2.00, 3.30, 4.10, 5.00]
V_blau = [0.000, 0.500, 1.000, 1.500, 2.00, 2.38, 2.42, 2.46, 2.51, 2.56, 2.63, 2.67, 2.70, 2.74, 2.82, 2.87, 2.91]
I_rot = [0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.50, 2.00, 3.30, 4.10, 5.00]
V_rot = [0.000, 0.000, 0.500, 1.000, 1.500, 1.520, 1.552, 1.582, 1.612, 1.640, 1.678, 1.702, 1.716, 1.736, 1.779, 1.802, 1.826]
I_gelb = [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.50, 2.00, 3.30, 4.10, 5.00]
V_gelb = [0.000, 0.000, 0.500, 1.000, 1.500, 1.621, 1.646, 1.674, 1.703, 1.730, 1.766, 1.790, 1.804, 1.824, 1.865, 1.886, 1.908]
iTot = [I_rot, I_gelb,  I_gruen, I_blau]
vTot = [V_rot, V_gelb,  V_gruen, V_blau]


# fitte hintergrund mit konstante
def const(x, c): return c  # Fitfunktion


# fitten
wl, rawData = background[1], background[6]
popt, pcov = curve_fit(const, wl, rawData)
avg = popt[0]  # subtrahiere von daten
stdDev = 1e-6*np.sqrt(pcov[0][0])*np.sqrt(len(wl))
print(stdDev)
# anzeigen
#plt.plot(wl, rawData, 'ro', markersize=1)
#plt.plot(wl, [avg]*len(wl), 'k-', lw=.6)
#plt.plot(wl, [avg+stdDev]*len(wl), 'k-', lw=.6)
#plt.plot(wl, [avg-stdDev]*len(wl), 'k-', lw=.6)
#plt.show()


def arrangedata(data, n):
    wl, dat = data[1], data[6] - avg
    found = False
    k, j = 0, 0
    for i, w in enumerate(wl):
        if dat[i] > n*stdDev and not found:
            k = i
            found = True
        elif found and not (dat[i] < n*stdDev and w>600):
            j = i+1
        elif found:
            j = i
            break
    return wl[k:j], dat[k:j]


def gaussian(x, a, b, c, d):
    return a*np.exp(-b*(x-c)**2) + d


def parabola(x, a, b, c):
    return a*(x-b)**2 + c


def lorentzian(x, a, b, c):
    return a / ((x**2 - b**2) + c**2 * x**2)


def planck(x, a, b, c):
    return (1/x**5)*(1/(np.exp(b/(x-c))-1))*a


def linear(x, a, b):
    return a*x+b


def cut(arr, compare, minimum, maximum):
    temp = []
    for i, x in enumerate(arr):
        if compare[i]<maximum and compare[i]>minimum:
            temp.append(x)
    return np.array(temp)


voltages = []

# bestimme spannungen
fig = plt.figure(figsize=(6, 4))
for j, i in enumerate(iTot):
    v = vTot[j]
    vN = cut(v, v, 1.5, 1e99)  # plotte nicht alle punkte die quasi auf x achse liegen
    iN = cut(i, v, 1.5, 1e99)
    iFit = cut(i, i, 2, 1e99)
    vFit = cut(v, i, 2, 1e99)
    print("Von der Kennlinie (Farbe: " + formatStrings[j] + ") werden " + str(len(iFit)) + " Elemente genommen um V(I=0) zu berechnen")
    popt, pcov = curve_fit(linear, vFit, iFit, sigma=[0.005]*len(vFit))
    a, b = ufloat(popt[0], np.sqrt(pcov[0][0])), ufloat(popt[1], np.sqrt(pcov[1][1]))
    v0 = -b/a; voltages.append(v0)  # spannungen sind uarray mit fehler aus dem linearem Fit
    vAll = np.linspace(nominal_value(v0), max(vFit), 1000)
    plt.plot(vAll, linear(vAll, *popt), formatStrings[j]+"-", lw=1.5)
    plt.plot([nominal_value(v0)], [0], 'kx', markersize=6)
    plt.plot(vN, iN, formatStrings[j]+"x", markersize=4)
    print("v0 (Farbe " + formatStrings[j] + ") = " + str(v0))
plt.xlabel(r'$U$ in $V$', fontsize=8)
plt.ylabel(r'$I$ in $mA$', fontsize=8)
plt.savefig('Graphen\\kennlinie_fit.pdf')
plt.close()


# bestimme die maximale wellenlänge aller spektren
estimationBounds = [
    [610, 650], [568.5, 610], [506, 545], [454, 487]
]
lambdaMax, lambdaErr = [], []
fig = plt.figure(figsize=(6, 4))


# fitte separat die Maxima aller Spektren mit Gauss
for i, name in enumerate(filenames):
    data = np.array(convert(name, ';', 78, "Data2"), dtype=float)
    wl, rawData = arrangedata(data, 1)
    rawData = rawData * 1e-6
    lower, upper = estimationBounds[i][0], estimationBounds[i][1]
    inverseWidth, middle, height, offset = 2/(upper-lower), (upper-lower)/2 + lower, 30000e-6, 0
    popt, pcov = curve_fit(gaussian, cut(wl, wl, lower, upper), cut(rawData, wl, lower, upper), p0=[height, inverseWidth, middle, offset])
    middle = popt[2]
    midUncert = ufloat(popt[2], np.sqrt(pcov[2][2]))
    print("LambdaMax (Farbe " + formatStrings[i] + ") = " + str(midUncert) + "nm")
    lambdaErr.append(std_dev(1/midUncert))
    lambdaMax.append(1/middle)

    wlFit = np.linspace(lower, upper, 1000)
    intensFit = gaussian(wlFit, *popt)
    ideal, data = gaussian(cut(wl, wl, lower, upper), *popt), cut(rawData, wl, lower, upper)
    redChisqu = np.sum(((ideal - data) / stdDev)**2)/(len(cut(wl, wl, lower, upper)) - 4)
    print("redChisqu (Farbe " + formatStrings[i] + ") = " + str(redChisqu) + " at dof = " + str(len(cut(wl, wl, lower, upper)) - 4))
    plt.plot(wlFit, 1e2*intensFit, formatStrings[i]+'--', lw=2)
    plt.plot(wl, 1e2*rawData, formatStrings[i]+"o", markersize=.5)
plt.xlabel(r'$\lambda$ in $nm$', fontsize=8)
plt.ylabel(r'$I(\lambda) d\lambda $ in b.E.', fontsize=8)
plt.savefig('Graphen\\maxFit.pdf')
plt.close()
print("Die geschätzten Fehler der inversen Wellenlängen sind: " + str(lambdaErr) + " nm^(-1)")


def totLinear(x, a):
    return a*x


# voltages = unumpy.uarray([1.826, 1.908, 3.07, 2.91], [0.001, 0.001, 0.001, 0.001])
# spannung abhängig von den inversen Wellenlängen - sollte linear sein:
popt, pcov = curve_fit(totLinear, lambdaMax, unumpy.nominal_values(voltages))
wlFit = np.linspace(min(lambdaMax), max(lambdaMax), 1000)
dof = len(lambdaMax) - 1
a = ufloat(popt[0], np.sqrt(pcov[0][0]))
redChisqu = np.sum(((unumpy.nominal_values(voltages) - totLinear(np.array(lambdaMax), *popt))  / (unumpy.std_devs(voltages)))**2) / dof
print("Steigung des letzten Graphen = " + str(a))
print("redchisqu = " + str(redChisqu))
scaleX, scaleY = con.c*1e9*1e-12, 1
fig = plt.figure(figsize=(6, 4))
plt.errorbar(scaleX*np.array(lambdaMax), scaleY*unumpy.nominal_values(voltages), yerr=scaleY*np.sqrt(redChisqu)*unumpy.std_devs(voltages), fmt='ro', linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
plt.plot(scaleX*np.array(lambdaMax), scaleY*unumpy.nominal_values(voltages), 'ko', markersize=2)
plt.plot(scaleX*wlFit, scaleY*totLinear(wlFit, *popt), 'k-', lw=.6)
plt.xlabel(r'$\frac{c}{\lambda} = \nu$ in $THz$', fontsize=8)
plt.ylabel(r'$U$ in $V$', fontsize=8)
plt.savefig('Graphen\\final.pdf')
plt.close()

print()
print("___PLANCKSCHES WIRKUNGSQUANTUM___")
print("h = " + str(a*1e-9*con.e/con.c))

