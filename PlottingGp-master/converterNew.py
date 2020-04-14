import csv
import os
import numpy as np
from uncertainties import unumpy
import sys
#convert nimmt ein txt file mit vorgegebener Struktur und gibt eine Matrix zurück
def convert(file_name,format,delimiter):
    parent = os.path.dirname(os.path.dirname(__file__))
    file = parent + "/PlottingGp-master/Rohdaten/" + file_name + "." + format

    #Multidimensionales array -> enthält am Ende alle Daten
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)
        row_count = sum(1 for i in csv_reader)
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        row0 = 0
        for rowNum, row in enumerate(csv_reader):
            if rowNum < row_count and rowNum>=row0:
                if rowNum == row0:
                    #also keine Tabs in den Überschriften verwenden!
                    column_count = len(row)
                    arr = [[] * 1 for i in range(column_count)]
                elif rowNum != 0:
                    for i in range(column_count):
                        arr[i].append(row[i].replace(',','.'))
                else:
                    arr = "File korrupt"
    return arr


#automates the calculation of meanValue of an uncertain array!
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