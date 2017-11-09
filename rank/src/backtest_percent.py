#Backtest analysis with two methods for the input data
import csv
import os
import sys
import re
import numpy as np
from operator import add

def backtest(input, period, portolios):
   #open file/take the data and transfer to matrix
   inputfile = input
   data = np.genfromtxt(inputfile,delimiter=',',names=True,dtype=None)

   #divide portfolio, calculate avg.return ad S.D. -------------------------------
   #Put date into date array
   date = []
   b = len(data['Date'])
   for i in range(0, b):
      a = data['Date'][i]
      if str(a) not in date:
       date.append(str(a))
	
   #Put return of same year, month in list
   c = len(date)
   ret = [[] for _ in xrange(c)]
   for i in range(0, c):
    for j in range(0, b):
     if data['Date'][j] == date[i]:
      ret[i].append(data['Next_Return'][j])

   #Slice to 10 parts and calculate avg.return according Retrun period (5->10)
   #avg_ret = return in corresponding month
   avg_ret = []
   port = portolios
   port = int(port)
   for i in range(0, c):
    ret_list = slice_list(ret[i],port)
    avg_ret.append([])
    avg_ret[i].append(ret_list)
   avg_ret = reduce(add, avg_ret)

   Profolio = [[] for _ in xrange(port)] 
   for i in range(0,port):
     for j in range(0, c):
      Profolio[i].append(avg_ret[j][i])

   #Calculate long/short of each month
   longshort = []
   for i in range(0, c):
    diff = np.mean(Profolio[0][i]) - np.mean(Profolio[port-1][i])
    longshort.append(diff)     
   
   Avg = [[] for _ in xrange(port)]
   S_D = [[] for _ in xrange(port)]
   Annual = [[] for _ in xrange(port)]
   longshort_avg = []
   longshort_sd = []
   longshort_annual = []
   rpt = period
   rpt = int(rpt)
   index = 0
   
   while ((c-index) > c%rpt):
    temp = [[] for _ in xrange(port)]
    for i in range(0,port):
     for j in range(0+index, rpt+index):
      temp[i].append(Profolio[i][j]) 
    #handling avg return and s.d.
    for i in range(0,port):
     #Avg[i].append(np.mean(np.array(reduce(add, temp[i]))))
     temp1 = []
     for j in range(0,rpt):
      temp1.append(np.mean(temp[i][j]))
     S_D[i].append(np.std(temp1))
     Avg[i].append(np.mean(temp1))
     Annual[i].append((np.sum(temp1)))#/(rpt/12.0))	 
  
    #long = np.mean(np.array(reduce(add, temp[0])))
    #short = np.mean(np.array(reduce(add, temp[port-1])))
    #longshort_avg.append(long - short)
    temp2 = []
    for i in range(0+index,rpt+index):
     temp2.append(longshort[i])
    longshort_sd.append(np.std(temp2))
    longshort_avg.append(np.mean(temp2))
    longshort_annual.append((np.sum(temp2)))#/(rpt/12.0))
	
    index += rpt 

   #Calculate total period
   total_avg = []
   total_sd = []
   total_annual = []
   for i in range (0,port):
    #list = reduce(add, Profolio[i])
    #total_avg.append(np.mean(np.array(list)))
    temp3 = []
    for j in range(0,c):
     temp3.append(np.mean(Profolio[i][j]))
    total_sd.append(np.std(temp3))
    total_avg.append(np.mean(temp3))
    total_annual.append((np.sum(temp3))/(c/rpt))#/(c/12.0))
   #total_long = np.mean(np.array(reduce(add, Profolio[0])))
   #total_short = np.mean(np.array(reduce(add, Profolio[port-1])))
   #total_lsavg = total_long - total_short
   total_lssd = np.std(longshort)
   total_lsavg = np.mean(longshort)
   total_lsan = (np.sum(longshort))/(c/rpt)#/(c/12.0)
   
   #Create the table
   report = [[] for _ in xrange(port+3)]
   loop = 0
   while (loop < (c/rpt)):
    report[0].append(str(rpt) + " month: ")
    report[0].append(date[0 + loop*rpt] + " to " + date[rpt-1 + loop*rpt])
    report[0].append('')
    report[0].append('')
    report[1].append('Prot')
    report[1].append('Avg Ret')
    report[1].append('St Dev')
    report[1].append('Annual Ret')
    for i in range (0,port):
     report[i+2].append(i+1)
     report[i+2].append(Avg[i][loop])
     report[i+2].append(S_D[i][loop])
     report[i+2].append(Annual[i][loop])
    report[port+2].append('Long/Short')
    report[port+2].append(longshort_avg[loop])
    report[port+2].append(longshort_sd[loop]) 
    report[port+2].append(longshort_annual[loop])	
    loop += 1
 
   #Total part
   report[0].append('Total Period: ' + date[0] + " to " + date[c-1])
   report[1].append('Port')
   report[1].append('Avg Ret')
   report[1].append('St Dev')
   report[1].append('Annual Ret')
   for i in range(0,port):
    report[i+2].append(i+1)
    report[i+2].append(total_avg[i])
    report[i+2].append(total_sd[i])
    report[i+2].append(total_annual[i])
   report[port+2].append('Long/Short')
   report[port+2].append(total_lsavg)
   report[port+2].append(total_lssd)
   report[port+2].append(total_lsan)
   
   return report
	 
def output(input, filename1, filename2):
   #Output file name, replace ".sorted" to ".report"
   #If no input output_file name, output input_file name
   if filename2 == None:
    outputfile = filename1.split(".")[0] + ".report" + ".csv"
   else: 
    outputfile = filename2.split(".")[0] + ".report" + ".csv"   
 
   with open(outputfile,"w") as newfile:       
	   wr = csv.writer(newfile)
	   for row in input:
	     wr.writerow(row)
		 
   newfile.close()

def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result