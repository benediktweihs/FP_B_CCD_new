import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import *


def funk(temper, a, b):
    return a*temper + b


def fit_funk(x, y, y_err):
    popt, pcov = curve_fit(funk, x, y, sigma=y_err, p0=[1.0, 1.0])
    x_values = np.linspace(x[0], x[-1], 10000)
    y_values = funk(x_values, *popt)
    error = []
    for i in range(len(popt)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)  #
        except:
            error.append(0.00)
    value = []
    rel_err = []
    k = 2*1.3806505 * 10 ** (-23) / (1.60217653 * 10 ** (-19))  # in eV umrechnen
    popt[0] = popt[0]*1000*k
    popt[1] = popt[1] * 1000 * k
    error[0] = error[0]*1000*k
    error[1] = error[1] * 1000 * k
    print('Werte: ', *popt)
    print('Fehler: ', *error)
    for i in range(len(popt)):
        value.append(popt[i])
        value.append(error[i])
        rel_err.append(abs(error[i]/popt[i]))
    print('rel. Fehler: ', *rel_err)
    print(' ')
    return x_values, y_values, value


if __name__ =='__main__':
    g = 2.1
    dg = 0.00001
    anz_px = 643123
    t = np.array([-21., -17., -13., -9., -5., -1., 3., 7., 11., 15., 19., 23., 27., 31.])+273.15
    m = np.array([100., 98., 106., 107., 120., 121., 129., 135., 135., 139., 144., 167., 220., 325.])*g
    # s_m = np.sqrt(m)
    s_m = np.array([17., 19., 22., 27., 35., 47., 63., 83., 110., 138., 162., 204., 265., 337.])*g
    for i in range(len(s_m)):
        s_m[i] = np.sqrt((s_m[i]/m[i])**2+(dg/g)**2)*m[i]
    for i in range(len(m)):
        s_m[i] = s_m[i]/m[i]
        m[i] = np.log(m[i]/(t[i]**(3/2)))
        s_m[i] *= m[i]
    t = 1000 / (np.array([-21., -17., -13., -9., -5., -1., 3., 7., 11., 15., 19., 23., 27., 31.]) + 273.15)
    # Achtung: division durch 1000. Steigung muss mit 1000 multipliziert werden!!!
    x_1, y_1, values_1 = fit_funk(t[0:11], m[0:11], s_m[0:11])
    x_2, y_2, values_2 = fit_funk(t[11:], m[11:], s_m[11:])
    fig = plt.figure()
    plt.grid()
    print(values_1)
    # chi_quadr = np.sum((funk(t, values_1[0], values_1[2]) - m) ** 2 / (s_m) ** 2) / (len(m) - 2)
    plt.errorbar(t, m, yerr=s_m, fmt='.', label='Datenpunkte')
    plt.plot(x_1, y_1, label='$y=-\\frac{(%.3f \pm %.3f)\,\mathrm{eV}}{k_\mathrm{B}}x+(%.2f \pm %.2f)$' %tuple(values_1))
    plt.plot(x_2, y_2, label='$y=-\\frac{(%.2f \pm %.2f)\,\mathrm{eV}}{k_\mathrm{B}}x+(%.1f \pm %.1f)$' %tuple(values_2))
    # plt.plot(x, y, label='fit: $a =%.2e\pm %.2e,$\n$ b=%.2e\pm %.2e,$\n$ c=%.2e\pm %.2e$' %tuple(values))
    plt.xlabel('1000/$T$ in K$^{-1}$')
    plt.ylabel('$N^e$')
    plt.tight_layout()
    plt.legend()
    plt.show()
