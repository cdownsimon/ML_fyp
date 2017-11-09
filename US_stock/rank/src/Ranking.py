#Rank the stocks in each month in descending order based on the Factor_For_Ranking
import csv
import os
import sys
import re
import datetime
import io

def rank(file_name):
   #open file
   file = open(file_name, 'rU')
   csvfile = csv.reader(file)

   #create a list to get data from the file
   table = []
   header = []
   ranking = []

   for row in csvfile:
    table.append(row)
   header = table[0]
   table.pop(0)

   # int the string
   for i in range(0,len(table)):
    ranking.append([])
    ranking[i].append(table[i][0].split("-")[0])
    ranking[i].append(table[i][0].split("-")[1])
    a=table[i][1]
    ranking[i].append(a)
    b=float(table[i][2])
    ranking[i].append(b)
    c=float(table[i][3])
    ranking[i].append(c)

   #Ranking according year, month, factor_for_ranking
   ranking.sort(key=lambda row:(row[0],row[1],-row[4]))

   #output new csv file
   for i in range(0,len(table)):
     y=ranking[i][0]
     m=ranking[i][1]
     #d=ranking[i][2]
     ranking[i][0] = str(y)+'-'+str(m)
   newranking = [[row[0],row[2],row[3],row[4]] for row in ranking]
   newheader = [header[0], header[1], header[2], header[3]]
   newranking.insert(0, newheader)
   
   file.close()
   return newranking


def output(file_name, input):
   #Output file name, replace ".data" to ".sorted"
   inputfile = file_name
   outputfile = inputfile.split(".")[0] + ".sorted" + ".csv"
   with open(outputfile,"w") as newfile:       
	   wr = csv.writer(newfile)
	   for row in input:
	     wr.writerow(row)
   newfile.close()
   
