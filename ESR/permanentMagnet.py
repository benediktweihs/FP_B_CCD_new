import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import *
import uncertainties.unumpy as unp
n = 7

def doStuff(delta):
    mu=ufloat(9.8e-24, 0.5e-24)  # J/T
    b=50*1e-6
    N=124
    R=0.15

    phi = np.array([0, 90, 180, 270], dtype=float)
    phi_err = np.array([1]*len(phi), dtype=float)
    phi -= delta
    phi_rad = phi*(np.pi/180)
    phi_uncert = unp.uarray(phi_rad, phi_err*(np.pi/180))
    I = np.array([1.12, 1.5, 1.96, 1.53], dtype=float)
    I_err = np.array([0.03]*len(I), dtype=float)
    I_uncert = unp.uarray(I, I_err)


    def B_helm(I_dc):
        return (8*const.mu_0*N*I_dc)/(np.sqrt(125)*R)


    def B_tot(i,g):
        return const.h*(g*const.e/(4*np.pi*const.m_e)*(b+(8*const.mu_0*N*i)/(np.sqrt(125)*R)))/mu
    #def B_tot(i):
    #    return (const.h/mu) *

    def B_p(B_tot, B_helm, phi_u):
        phi = nominal_value(phi_u)
        #if -np.pi/100<phi<1*np.pi/100: return -B_helm + B_tot
        #if .9*np.pi/2<phi<1.1*np.pi/2: return -unp.cos(phi_u)*B_helm + (B_tot**2 - B_helm**2)**(1/2)
        if np.pi*.9<phi<np.pi*1.1: return -B_helm+B_tot
        #if .9*3*np.pi/2<phi<1.1*3*np.pi/2: return -unp.cos(phi_u)*B_helm + (B_tot**2 - B_helm**2)**(1/2)
        #if phi == 0: return -B_helm * np.cos(phi) + (B_tot**2 - np.sin(phi)**2 * B_helm**2)**(1/2)
        return -B_helm * unp.cos(phi_u) + (B_tot**2 - unp.sin(phi_u)**2 * B_helm**2)**(1/2)


    B_tot_uncert = 1e3*B_tot(I_uncert, ufloat(1.979, 0.019))
    BHelmUncert = 1e3*B_helm(I_uncert)
    #print(str((B_tot_uncert / const.h)*mu))
    dof = 4 - 1


    # fit with constant
    B_p_uncert = [B_p(B_tot_uncert[i], BHelmUncert[i], r) for i, r in enumerate(phi_uncert)]
    def cons(x, c): return c


    popt, pcov = curve_fit(cons, phi, unp.nominal_values(B_p_uncert), sigma=unp.std_devs(B_p_uncert))
    chisqu = np.sum(((cons(phi, *popt) - unp.nominal_values(B_p_uncert))/unp.std_devs(B_p_uncert))**2) / dof
    #print(chisqu)
    print(np.sqrt(pcov[0][0]))
    plt.plot(phi, unp.nominal_values(B_p_uncert), 'ko', markersize=3)
    plt.errorbar(phi, unp.nominal_values(B_p_uncert), yerr=np.sqrt(chisqu)*unp.std_devs(B_p_uncert), fmt='ro', linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
    plt.plot(phi, [popt[0]]*len(phi), 'k-', lw=1)
    #plt.show()
    #plt.close()
    return chisqu, delta, popt[0]

'''
chisqu = 1e99
deltaFin=0
Bfin = 0
for delta in range(-180,0):
    temp, deltaTemp, Btemp = doStuff(delta)
    if temp < chisqu:
        Bfin = Btemp
        chisqu = temp
        deltaFin = deltaTemp
    plt.close()'''


chisqu, deltaFin, Bfin = doStuff(0)
print(Bfin)
print(chisqu)
print(deltaFin)
doStuff(deltaFin)
plt.show()