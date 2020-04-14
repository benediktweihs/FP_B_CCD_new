import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import *


def funk(b, temper):
    return b[0]*temper**(3/2)*np.exp(b[1]/(2*temper))+b[2]*temper**3*np.exp(b[1]/temper)

def fit_funk(x, y, y_err):
    mydata = RealData(x, y, sy=y_err)
    linear = Model(funk)
    myodr = ODR(mydata, linear, beta0=[1., 1., 1.])
    myout = myodr.run()
    x_fit = np.linspace(x[0], x[-1], 10000)  #
    y_fit = funk(myout.beta, x_fit)
    myout.pprint()
    return x_fit, y_fit


if __name__ =='__main__':
    g = 1
    t = np.array([-21, -17, -13, -9, -5, -1, 3, 7, 11, 15, 19, 23, 27, 31]) + 273.15
    m = np.array([100., 98., 106., 107., 120., 121., 129., 135., 135., 139., 144., 167., 220., 325.]) * g
    s_m = np.array([17., 19., 22., 27., 35., 47., 63., 83., 110., 138., 162., 204., 265., 337.]) * g
    print(s_m)
    x, y = fit_funk(t, m, s_m)
    fig = plt.figure()
    plt.grid()
    plt.errorbar(t, m, yerr=s_m, fmt='.')
    plt.plot(x, y)
    plt.xlabel('$T$ in K')
    plt.ylabel('$N^e$')
    plt.legend()
    plt.show()
