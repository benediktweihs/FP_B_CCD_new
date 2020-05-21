import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from Emissionsspektroskopie.converter import *
from numpy import *
from uncertainties import unumpy
from matplotlib import rc

rc('text', usetex=True)  # Latex

def cutWl(wl):
    newWl = []
    found = False
    k, j = 0, 0
    for i, w in enumerate(wl):
        if w>450 and w<790:
            newWl.append(w)
            if not found: k = i
            found = True
        else:
            if found: j = i
            found = False
    return np.array(newWl, dtype=float), k, j


def const(x, c):
    return c


directory = "C:\\Users\\weihs\\PycharmProjects\\FP_B_CCD_new\\Emissionsspektroskopie\\Data\\"
rawData0 = np.array(convert("gelb.txt", ';', 78, "Data"), dtype=float)[6]
wl0 = np.array(convert("gelb.txt", ';', 78, "Data"), dtype=float)[1]
voltages = np.array([
    1.707, 1.741, 1.766, 1.793, 1.81, 1.85, 1.87, 1.92, 1.97, 2.03, 2.09, 2.16, 2.23, 2.31
], dtype=float)


# fehler aus hintergrundsignal
wl0, lower, upper = cutWl(wl0)
rawData0fit = rawData0[lower:upper]
popt, pcov = curve_fit(const, wl0, rawData0fit, p0=[1300])
avg, stdDev = popt[0], np.sqrt(pcov[0][0])*np.sqrt(len(wl0))
print(stdDev)
print(np.sqrt(avg))
fig = plt.figure(figsize=(6, 4))
plt.xlabel(r'$\lambda$ in $nm$', fontsize=8)
plt.ylabel(r'$I(\lambda) d\lambda$ in b.E.', fontsize=8)
plt.plot(wl0, rawData0fit, 'ro', markersize=1)
plt.plot(wl0, [avg]*len(wl0), 'k-', lw=.5)
plt.plot(wl0, [avg+stdDev]*len(wl0), 'b-', lw=.5)
plt.plot(wl0, [avg-stdDev]*len(wl0), 'b-', lw=.5)
plt.savefig('Graphen\\background.pdf')
plt.close()


def cutNew(wl, dat, n):
    newWl = []
    found = False
    k, j = 0, 0
    for i, w in enumerate(wl):
        if dat[i] > n*stdDev and not found:
            k = i
            found = True
            newWl.append(w)
        elif found and not (dat[i] < n*stdDev and w>600):
            newWl.append(w)
            j = i+1
        elif found:
            j = i
            break
    return np.array(newWl, dtype=float), k, j


def cut(data, n):
    wlFirst, sth, sth1 = cutWl(data[1])
    rawDataFirst = (data[6] - rawData0)[sth:sth1]
    wl, lower, upper = cutNew(wlFirst, rawDataFirst, n)  # wavelength
    rawData = rawDataFirst[lower:upper]
    return wl, rawData


intensities = []


for i, filename in enumerate(os.listdir(directory)):
    print(filename)
    if i == 0: continue  # Hintergrund schon in rawData0
    data = np.array(convert(filename, ';', 78, "Data"), dtype=float)
    #if i < 3: wl, rawData = cut(data, 3)
    wl, rawData = cut(data, 3)
    delta = np.array([-wl[i-1]+wl[i] for i in range(1, len(wl))], dtype=float)
    #rawDataUncert = unumpy.uarray(rawData, stdDev)
    intIntens = sum([min(rawData[i+1], rawData[i])*dw*1e-6 for i, dw in enumerate(delta)])
    intensities.append(intIntens)
    # Plots für gegebene Stromstärken
    plt.plot(wl, rawData, 'ro', markersize=.5)
    plt.show()


def exp(x, a, c, v):
    return a*np.exp(c*x) + v


popt, pcov = curve_fit(exp, voltages, intensities, p0=[.1,1,0])  # falls keine uarrays
#popt, pcov = curve_fit(parabola, voltages, unumpy.nominal_values(intensities), p0=[.1,1,0])  # mit uarrays
vFit = np.linspace(voltages[0], voltages[-1], 1000)

fig = plt.figure(figsize=(6, 4))
plt.plot(vFit, exp(vFit, *popt), 'k-', lw=.5)
plt.xlabel(r'$U$ in $V$', fontsize=8)
plt.ylabel(r'$I_{int} (\lambda)$ in b.E.', fontsize=8)
plt.plot(voltages, intensities, 'bo', markersize=4)
#print(unumpy.std_devs(intensities))
#plt.errorbar(voltages, unumpy.nominal_values(intensities), yerr=unumpy.std_devs(intensities), fmt='ro', linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
plt.savefig('Graphen\\totIntens.pdf')