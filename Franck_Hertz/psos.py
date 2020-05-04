from converterNew import *
import numpy as np
import matplotlib.pyplot as plt

times1 = np.array(convert("times1", "txt", ','), dtype=float)[0]
times2 = np.array(convert("times2", "txt", ','), dtype=float)[0]
t1 = []
t2 = []
old = 0
sum1 = 0
sum2 = 0
for i in range(len(times1)//30):
    sum1, sum2 = 0, 0
    for l in range(30):
        sum1 = sum1 + times1[i*l]
        sum2 = sum2 + times2[i*l]
    t1.append(sum1)
    t2.append(sum2)
t1 = np.array(t1)
t2 = np.array(t2)
plt.plot(t1,'ro')
plt.plot(t2,'bo')
plt.savefig("/home/benedikt/PycharmProjects/FP_B_CCD_Patrick/Franck_Hertz/Data/time.pdf")
