import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import *


def funk(temper, a, b, c):
    # return a*temper**(3/2)*np.exp(-b/(2*temper))  #
    k = 1.3806505*10**(-23)/(1.60217653*10**(-19))
    return (a*temper**(3/2)*np.exp(-b/(2*k*temper))+c*temper**(3)*np.exp(-b/(k*temper)))


def fit_funk(x, y, y_err):
    popt, pcov = curve_fit(funk, x, y, sigma=y_err, p0=[1.0, -0.1, 1.0])
    print('Werte: ', *popt)
    x_values = np.linspace(x[0], x[-1], 10000)
    y_values = funk(x_values, *popt)
    error = []
    for i in range(len(popt)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append(0.00)
    value = []
    rel_err = []
    print('Fehler: ', *error)
    for i in range(len(popt)):
        value.append(popt[i])
        value.append(error[i])
        rel_err.append(abs(error[i]/popt[i]))
    print('rel. Fehler: ', *rel_err)
    return x_values, y_values, value


if __name__ =='__main__':
    g = 2.1
    dg = 0.00001
    t = np.array([-21, -17, -13, -9, -5, -1, 3, 7, 11, 15, 19, 23, 27, 31])+273.15
    m = np.array([100., 98., 106., 107., 120., 121., 129., 135., 135., 139., 144., 167., 220., 325.]) * g
    s_m = np.array([17., 19., 22., 27., 35., 47., 63., 83., 110., 138., 162., 204., 265., 337.])*g
    for i in range(len(s_m)):
        s_m[i] = np.sqrt((s_m[i]/m[i])**2+(dg/g)**2)*m[i]
    x, y, values = fit_funk(t, m, s_m)
    fig = plt.figure()
    plt.grid()
    plt.errorbar(t, m, yerr=s_m, fmt='.', label='Datenpunkte mit Fehler')
    values_out = values
    values_out.append(values[2])
    values_out.append(values[3])
    plt.plot(x, y, label='$y=(%.2f\pm%.2f\,\\frac{\mathrm{A}}{K^{3/2}})x^{\\frac{3}{2}}exp(-\\frac{%.3f \pm%.3f\,\mathrm{eV}}'
                         '{2k_\mathrm{B}x})$\n  + $(%.1e\pm%.1e\,\\frac{\mathrm{A}}{K^{3/2}})x^3exp(-\\frac{%.3f \pm%.3f'
                         '\,\mathrm{eV}}{k_\mathrm{B}x})$' %tuple(values))
    # plt.plot(x, y, label='fit: $a =%.2e\pm %.2e,$\n$ b=%.2e\pm %.2e,$\n$ c=%.2e\pm %.2e$' %tuple(values))
    plt.xlabel('$T$ in K')
    plt.ylabel('$N^e$')
    plt.legend(loc=2, fancybox=True, fontsize='large')
    plt.tight_layout()
    plt.show()
