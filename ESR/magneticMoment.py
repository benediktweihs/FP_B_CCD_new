from uncertainties import *
from uncertainties import unumpy
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
import converterNew as conv
from scipy.optimize import curve_fit
from matplotlib import rc
from scipy import constants as const
import scipy.odr.odrpack as odrpack
rc('text', usetex=True)
import glob, os

freq=np.array([44,41.5,38,36.7,35.2,33.2,30.9,29.35,28.1])*1e6#Mhz
d_freq=np.array([0.5,0.2,0.5,0.5,0.5,0.5,0.3,0.35,0.4])*1e6
current=np.array([2.04,1.92,1.77,1.70,1.63,1.61,1.50,1.36,1.28])#A
d_curr=np.ones_like(current)*0.03

f=unumpy.uarray(freq,d_freq)
I=unumpy.uarray(current,d_curr)
freq=unumpy.nominal_values(f)
current=unumpy.nominal_values(I)
k=f/I
nom_k=unumpy.nominal_values(k)
d_k=unumpy.std_devs(k)
N=124
R=0.15
dr=1e-3
g=2
mu_b=const.physical_constants['Bohr magneton'][0]
RHS=8*const.mu_0*mu_b*124/const.h/np.sqrt(125)/R*g

b=50*1e-6
def func(i,g):
    return g*const.e/(4*np.pi*const.m_e)*(b+(8*const.mu_0*N*i)/(np.sqrt(125)*R))
#def func(wavelength,k,o):
#    return o+k*wavelength
popt,pcov = curve_fit(func,current,freq,sigma=d_freq,absolute_sigma=True)
perr=np.sqrt(np.diag(pcov))
print('g_alt=',popt[0])#,'\tb_alt=',popt[1])
y_err=abs(func(current,*popt)-func(current+d_curr,*popt))+d_freq

popt,pcov = curve_fit(func,current,freq,sigma=y_err,absolute_sigma=True)
perr=np.sqrt(np.diag(pcov))
g_m=ufloat(popt[0],perr[0])
b_m=0#ufloat(popt[1],perr[1])
print('g=',g_m,'\tb=',b_m)

def func_p(i,g,b):
    return g*const.e/(4*np.pi*const.m_e)*(b+(8*const.mu_0*N*i)/(np.sqrt(125)*(R+dr)))
def func_m(i,g,b):
    return g*const.e/(4*np.pi*const.m_e)*(b+(8*const.mu_0*N*i)/(np.sqrt(125)*(R-dr)))
popt_p,pcov = curve_fit(func_p,current,freq,sigma=y_err,absolute_sigma=True)
popt_m,pcov = curve_fit(func_m,current,freq,sigma=y_err,absolute_sigma=True)
g_sys,b_sys=(popt_p-popt_m)/2#abs(func_p(current,*popt_p)-func_m(current+d_curr,*popt_m))
print('syst error=',g_sys,b_sys)

result = curve_fit(func_m,current,freq,sigma=y_err,absolute_sigma=True,full_output=True)
s_sq = (result[2]['fvec']**2).sum()/(len(result[2]['fvec'])-len(result[0]))
print('chi2=',s_sq)
def f(B, i):
    return B[0]*const.e/(4*np.pi*const.m_e)*(B[1]+(8*const.mu_0*N*i)/(np.sqrt(125)*R))

fit = odrpack.Model(f)
mydata1 = odrpack.RealData(current, freq, sx=d_curr, sy=d_freq)
myodr1 = odrpack.ODR(mydata1, fit, beta0=[1., 2.])
myoutput1 = myodr1.run()
myoutput1.pprint()
print(myoutput1.res_var)
g=myoutput1.beta[0]
B_earth=myoutput1.beta[1]
dg=myoutput1.sd_beta[0]
dB_earth=myoutput1.sd_beta[1]

gfaktor=ufloat(g,dg)
gyro=gfaktor*const.e/2/const.m_e
gyro_m=g_m*const.e/2/const.m_e
gyro_sys=g_sys*const.e/2/const.m_e
mu=gyro*const.hbar/2
mu_m=gyro_m*const.hbar/2
mu_sys=gyro_sys*const.hbar/2

print('mu=',mu,'\t mu_m',mu_m)
print('syst error=',mu_sys)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(current,freq*1e-6,'ko',label='data')
ax.errorbar(current,freq*1e-6,yerr=y_err*1e-6,marker='o',linestyle='',linewidth=1,markersize=0,capsize=2,color='black')
#ax.plot(current,f([g,B_earth],current)*1e-6,label='linear fit (x,y-errors)')
ax.plot(current,func(current,*popt)*1e-6,'k--',label='linear fit')
print("______________POPT______________")
print(str(popt))
ax.set_xlabel('Current in A')
ax.set_ylabel('Frquency in MHz')
ax.legend(loc='best')
plt.savefig('plot.pdf')
plt.show()

print("___cur___")
print(str(current))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(current,freq*1e-6,'ko',label='data')
ax.errorbar(current,freq*1e-6,yerr=d_freq*1e-6,xerr=d_curr,marker='o',linestyle='',linewidth=1,markersize=0,capsize=2,color='black')
ax.plot(current,f([g,B_earth],current)*1e-6,label='linear fit (x,y-errors)')
#ax.plot(current,f(current,*popt)*1e-6,'k--',label='linear fit')
ax.set_xlabel('Current in A')
ax.set_ylabel('Frquency in MHz')
ax.legend(loc='best')
plt.savefig('plot_addit.pdf')
plt.show()
