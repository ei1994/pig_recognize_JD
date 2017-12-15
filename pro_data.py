# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:42:47 2017

@author: DELL
"""
import copy
import numpy as np

offset = 0.0001

with open('out_v4_13.csv','r') as f:
  content =[line.strip() for line in f.readlines()]

all_content = []

for i in range(0,len(content),30):
  subs = content[i:i+30]
  x = []
  for sub in subs:
    temp = sub.split(',')[-1]
    x.append(temp)

  prob = list(map(float,x))
  prob_max = max(prob)
  index = prob.index(prob_max)
  result = copy.copy(prob)
  
  if prob_max < 0.3:
    all_sum = 0
    a = np.array(prob)
    index_a = np.argsort(a)
    index_b = index_a[:27]
    
    for m in (index_b):
      all_sum = all_sum + a[m]
    abc = all_sum/27
    for j in index_b:
      
      result[j] = abc


    temps = []
      
    for i in range(len(subs)):
      temp1 = subs[i].split(',')[0]+ ','+subs[i].split(',')[1]+',' +str(result[i])
      temps.append(temp1)
      
    all_content.extend(temps)

  else:
    all_content.extend(subs)
#
with open('new1.csv','w') as f1:
  for row in all_content:
    f1.writelines(row+'\n')
   
    
    
    
    
    