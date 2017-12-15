# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:54:37 2017

@author: DELL
"""

import copy
import numpy as np

offset = 0.0001
q = []
pq = 0

with open('out_b_19_21.csv','r') as f:
  content =[line.strip() for line in f.readlines()]
  
with open('out_b_19_face.csv','r') as f:
  content1 =[line.strip() for line in f.readlines()]

all_content = []

for i in range(0,len(content),30):
      subs = content[i:i+30]
      x = []
      for sub in subs:
        temp = sub.split(',')[-1]
        x.append(temp)
    
      prob = list(map(float,x))
      prob_max = max(prob)
    #  index = prob.index(prob_max)
    #  result = copy.copy(prob)
    #  a = np.array(prob)
    #  index_a = np.argsort(-a)
      
      if  0.3>prob_max > 0 :
        number = subs[1].split(',')[0]
        q.append(number)
#        pq = pq + 1
        for i1 in range(0,len(content1),30):
          subs1 = content1[i1:i1+30]
          y = []
          number1 = subs1[1].split(',')[0]
          if (number1 == number):
              for sub in subs1: 
                temp = sub.split(',')[-1]
                y.append(temp)
                prob1 = list(map(float,y))
                prob_max1 = max(prob1)
          else:
              continue
        temps = []
          
    #      for j in range(len(subs)):
    #          temp1 = subs[j].split(',')[0]+ ','+subs[j].split(',')[1]+',' + str(prob1[j])
    #          temps.append(temp1)
    
        if prob_max1 > prob_max and prob_max1 > 0.7 :
              pq = pq + 1
              for i in range(len(subs)):
    #              prob1[i] = prob1[i]/1.000001
                  temp1 = subs[i].split(',')[0]+ ','+subs[i].split(',')[1]+',' + str(prob1[i])
                  temps.append(temp1)
        else:
              for i in range(len(subs)):
                  temp1 = subs[i].split(',')[0]+ ','+subs[i].split(',')[1]+',' + str(prob[i])
                  temps.append(temp1)
    
        all_content.extend(temps)
      else:
        all_content.extend(subs)

with open('out_b_19_22.csv','w') as f1:
  for row in all_content:
    f1.writelines(row+'\n')

    
    
    
    
    