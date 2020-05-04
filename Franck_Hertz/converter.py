import csv
import os
import numpy as np
from uncertainties import unumpy
import sys
#convert nimmt ein txt file mit vorgegebener Struktur und gibt eine Matrix zurück
def convert(file_name,delimiter,row0):
    parent = os.path.dirname(os.path.dirname(__file__))
    file = parent + "/Franck_Hertz/Data/" + file_name

    #Multidimensionales array -> enthält am Ende alle Daten
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)
        row_count = sum(1 for i in csv_reader)
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for rowNum, row in enumerate(csv_reader):
            if rowNum < row_count-3 and rowNum>=row0:
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