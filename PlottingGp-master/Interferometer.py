from __future__ import unicode_literals
from uncertainties import *
from converterNew import *
from uncertainties import unumpy
from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

import scipy.odr.odrpack as odrpack


##matplotlib.rcParams['text.usetex'] = True
##matplotlib.rcParams['text.latex.unicode'] = True
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#alle Graphen werden in Graphen gespeichert
if not os.path.exists("Graphen"):
    os.mkdir("Graphen")


def meanValue(uncertainArray):
    x = unumpy.nominal_values(uncertainArray)
    y = unumpy.std_devs(uncertainArray)

    g = [1/(i**2) for i in y]
    gewicht = 1/sum(g)
    partial = 0
    for i in range(len(y)):
        partial = partial + x[i]/(y[i]**2)
    nominal = gewicht * partial
    stdDev = np.sqrt(gewicht)
    return [nominal, stdDev]

#data = convert("test")


#fehlerauswertung
#timef = np.array(dataf[0],dtype=float)  #ms

#timef=timef[0:500:]


###################
#n*lambda/2  mbar
p=[]
p1=[]
p2=[]
pu=ufloat(960.16,0.2)
L=ufloat(632.8*1e-9,1*1e-9)  #lambda 0
l=ufloat(38.3*1e-3,1*1e-3)   #länge gefäß
#d=[]    # wegunterschied
#p0=[650,620,590,560,520,500,470,430,400,370,340,290,240,210,160,120,90,60]
p0=[640,610,580,555,520,490,460,430,400,370,340,310,270,245,210,180,145,110,70]
#p0=[650,620,590,560,520,500,470,430,400,370,340,300,270,240,210,170,130,95,70]
p1=[620,585,550,520,485,460,430,400,365,340,300,278,240,210,180,140,105,80,0]
p2=[625,590,560,535,500,460,430,400,380,325,300,280,240,200,180,145,115,80,0]
R1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.5]
R2=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20.5]
dn1=[]    #delta n
dn2=[]    #delta n
R=[]    #Ringe für druck
s=[]
dn=[]    #delta n




w_minus=[]
w_plus=[]
alpha=[]
alpha_grad=[]
Ringe_plus=[1,3,6,7.5,9,13,14,19,19,21]
Ringe_minus=[1,3,5,9,9,11,14,17,19,20]
R_p=[]
R_m=[]
n_glas_p=[]
n_glas_m=[]
n_glas_mittel=0.0
d=ufloat(5.6*1e-3,0.02*1e-3)   #dicke glas


for i in range(len(p0)):
    p0[i]=(pu-ufloat(p0[i],10))
    R.append(ufloat(i+1,0.5))
    dn.append(R[i]*L/2/l)

for i in range(len(p1)):
    p1[i]=(pu-ufloat(p1[i],10))
    p2[i] =(pu - ufloat(p2[i], 10))
    R1[i]=(ufloat(R1[i], 0.5))
    R2[i] = (ufloat(R2[i], 0.5))
    dn1.append(R1[i]*L/2/l)
    dn2.append(R2[i] * L / 2 / l)

#p0=p0[1:len(p0):]
#dn=dn[1:len(dn):]
#p1=p1[1:len(p1):]
#dn1=dn1[1:len(dn1):]
#p2=p2[1:len(p2):]
#dn2=dn2[1:len(dn2):]

for i in range(len(Ringe_minus)):
    if i<5:
        Ringe_plus[i]=(ufloat(Ringe_plus[i],0.5))
        Ringe_minus[i] = (ufloat(Ringe_minus[i], 0.5))
    else:
        Ringe_plus[i]=(ufloat(Ringe_plus[i],1))
        Ringe_minus[i] = (ufloat(Ringe_minus[i], 1))
    if i>0:
        R_p.append(R_p[i-1]+Ringe_plus[i])
        R_m.append(Ringe_minus[i] + R_m[i - 1])
    else:
        R_p.append(Ringe_plus[i])
        R_m.append(Ringe_minus[i])
    alpha.append(ufloat((i + 1)*np.pi/180, 0.5*np.pi/180))
    #alpha_grad.append(ufloat((i + 1) , 1 ))
    n_glas_p.append(1/(1-(R_p[i]*L)/(d*alpha[i]*alpha[i])))
    n_glas_m.append(1/(1-(R_m[i]*L)/(d*alpha[i]*alpha[i])))
    n_glas_mittel +=(n_glas_m[i]+n_glas_p[i])/2
n_glas=[]
for i in range(len(n_glas_p)*2):
    if i <len(n_glas_p):
        n_glas.append(n_glas_m[i])
    else:
        n_glas.append(n_glas_p[i-len(n_glas_p)])

    print(std_dev(n_glas[i]))
    #print(n_glas_p[i])

print("gewichteter Mittelwert n_Glas")
#print(n_glas_p)
#print(R_p)
#print(n_glas_mittel/10)

print(np.mean(np.array(np.concatenate((unumpy.nominal_values(n_glas_p),unumpy.nominal_values(n_glas_m))))))
print(np.std(np.array(np.concatenate((unumpy.nominal_values(n_glas_p),unumpy.nominal_values(n_glas_m))))))

print(meanValue(n_glas))

#print(alpha_grad)

#################
druck=[unumpy.nominal_values(p0),unumpy.nominal_values(p1),unumpy.nominal_values(p2)]
Er_druck=[unumpy.std_devs(p0),unumpy.std_devs(p1),unumpy.std_devs(p2)]

#druck[1]=[1:5:]

brechungsindex=[unumpy.nominal_values(dn),unumpy.nominal_values(dn1),unumpy.nominal_values(dn2)]
Er_brech=[unumpy.std_devs(dn),unumpy.std_devs(dn1),unumpy.std_devs(dn2)]

Maxima=[unumpy.nominal_values(R),unumpy.nominal_values(R1),unumpy.nominal_values(R2)]
Er_max=[unumpy.std_devs(R),unumpy.std_devs(R1),unumpy.std_devs(R2)]


linfit1=np.polyfit(druck[0],brechungsindex[0],1,w=Er_druck[0])
print(linfit1)
linfit2=np.polyfit(druck[1],brechungsindex[1],1)
print(linfit2)
linfit3=np.polyfit(druck[2],brechungsindex[2],1)
print(linfit3)


#[ 2.34733290e-07 -6.08446455e-05]
#[ 2.62854976e-07 -8.24342656e-05]
#[ 2.58426791e-07 -7.77971621e-05]


def f(B, x):
    return B[0]*x + B[1]

linear = odrpack.Model(f)

#for i in range(0):
mydata1 = odrpack.RealData(druck[0], brechungsindex[0], sx=Er_druck[0], sy=Er_brech[0])
mydata2 = odrpack.RealData(druck[1], brechungsindex[1], sx=Er_druck[1], sy=Er_brech[1])
mydata3 = odrpack.RealData(druck[2], brechungsindex[2], sx=Er_druck[2], sy=Er_brech[2])
myodr1 = odrpack.ODR(mydata1, linear, beta0=[1., 2.])
myodr2 = odrpack.ODR(mydata2, linear, beta0=[1., 2.])
myodr3 = odrpack.ODR(mydata3, linear, beta0=[1., 2.])
myoutput1 = myodr1.run()
myoutput2 = myodr2.run()
myoutput3 = myodr3.run()
myoutput1.pprint()


#print(myoutput1.beta[0])
linfit1 = ufloat(myoutput1.beta[0],myoutput1.sd_beta[0])
linfit2 = ufloat(myoutput2.beta[0],myoutput2.sd_beta[0])
linfit3 = ufloat(myoutput3.beta[0],myoutput3.sd_beta[0])
offset1=myoutput1.beta[1]
offset2=myoutput2.beta[1]
offset3=myoutput3.beta[1]

print((linfit1+linfit2+linfit3)/3)

#print(1+linfit1[0]*pu)
print(1+linfit1*pu)
print(1+linfit2*pu)
print(1+linfit3*pu)

print("Mittel:")
print(1+(linfit1+linfit2+linfit3)/3.0*pu)

#k = sum(druck[0] * (brechungsindex[0])) / sum(druck[0]**2)
##print(sum(brechungsindex[0]))
##print(druck[1][5])
#x=[0.0]
#y=[0.0]
#for i in range(len(p0)):
#    x.append(druck[0][i])
#    y.append(druck[0][i]*k)

from numpy import exp, linspace, random
x = linspace(0, 1000, 10)



def mass(A,z=[]):

    e=1.6021766208 *1e-19
    mn=1.674927471*1e-27
    mp=1.672621898 *1e-27
    me=9.10938356 *1e-31
    av=15.58*1e6 *e
    As= 16.91*1e6 *e
    ac=0.71*1e6 *e
    aa=23.21*1e6 *e
    ap = 11.46 * 1e6 * e
    c=299792458
    m=[]
    u = 1.660539040 * 1e-27
    for i in range(len(z)):
        m.append(((A-z[i]) * mn + z[i] * mp + z[i] * me - (av * A - As * A**(2/3) - ac * z[i]**2 * A**(-1/3) - aa * (A-z[i]*2)**2 / A + 0 * ap*A**(-1/2))/(c**2))/u)
    return m

# z1=[9,10,11,12,13,14,15]
# z2=[47,48,49,50,51,52,53]
# m1=mass(25,z1)
# m2=mass(118,z2)
# u=1.660539040*1e-27
# mg1=[25.012199229,24.997788707,24.989953969,24.985836976,24.990428102,25.004108808,25.02119]
# mg2=[117.914595487,117.906921869,117.906356616,117.90160657,117.905532139,117.905853629,117.913074]
# print(m1)
# print(m2)
# Abbidlungen
fig, ax = plt.subplots()
#ax.plot(z2,m2, color = 'red', marker='o', linestyle='', markersize = 5,label=r'$U_{Bat}$')
#ax.plot(z2,mg2, color = 'black', marker='o', linestyle='', markersize = 5,label=r'$U_{Bat}$')
#ax.plot(Wegunterschied*1e9,druck, color = 'red', marker='o', linestyle='', markersize = 1,label=r'$U_{Bat}$')
ax.plot(druck[0],brechungsindex[0], color = 'red', marker='o', linestyle='', markersize = 3,label=r'$1. Messreihe$')
ax.plot(druck[1],brechungsindex[1], color = 'blue', marker='o', linestyle='', markersize = 3,label=r'$2. Messreihe$')
ax.plot(druck[2],brechungsindex[2], color = 'green', marker='o', linestyle='', markersize = 3,label=r'$3. Messreihe$')
ax.plot(x,unumpy.nominal_values(linfit1)*x+offset1, color = 'red', marker='', linestyle='--',linewidth='0.5', markersize = 1)
ax.plot(x,unumpy.nominal_values(linfit2)*x+offset2, color = 'blue', marker='', linestyle='--', linewidth='0.5',markersize = 1)
ax.plot(x,unumpy.nominal_values(linfit3)*x+offset3, color = 'green', marker='', linestyle='--', linewidth='0.5',markersize = 1)
#ax.plot(x,y, color = 'black', marker='', linestyle='-', markersize = 1,label=r'$Brechungsindex (n-1)}$')
ax.errorbar(druck[0], brechungsindex[0], yerr=Er_brech[0], xerr=Er_druck[0], fmt='ko', linewidth=0.8, capsize=1, capthick=0.8, markersize=1)
ax.errorbar(druck[1], brechungsindex[1], yerr=Er_brech[1], xerr=Er_druck[1], fmt='ko', linewidth=0.8, capsize=1, capthick=0.8, markersize=1)
ax.errorbar(druck[2], brechungsindex[2], yerr=Er_brech[2], xerr=Er_druck[2], fmt='ko', linewidth=0.8, capsize=1, capthick=0.8, markersize=1)
#ax.plot(time,Ubat-Ufuse,color = 'green', marker='o', linestyle='',markersize = 1,label=r'$U_{ConductivePath}$')
ax.legend(loc='upper left',frameon=True)

#axarr[2].legend(loc='lower right', frameon=True)

#ax.set(xlabel='Z',ylabel='M')
ax.set(xlabel='Druck (mbar)',ylabel='Brechungsindex (n-1)')
ax.grid(True,linestyle='--',linewidth = 0.3)
plt.xlim(250,1000)
plt.ylim(0,0.00020)

#plt.hist(Channel0f, bins=5)


#plt.xscale('log')
#plt.title('test')

#plt.errorbar(X, Y, yerr=0.1, fmt='ko', linewidth=0.8, capsize=3, capthick=0.8, markersize=5)

#i='Voltage'
#plt.savefig("Graphen/Temperature_" + str(i) + ".png")

plt.show()