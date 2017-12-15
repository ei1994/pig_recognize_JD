


with open('out_v4_13.csv','r') as f:
  content =[line.strip() for line in f.readlines()]

all_content = []
for i in range(0,len(content),30):
  subs = content[i:i+30]
  prob = []
  for sub in subs:
    temp = sub.split(',')[-1]
    prob.append(temp)

  prob1 = list(map(float,prob))

  if max(prob1) >0.8:
    index = prob1.index(max(prob1))

    prob2 =[ 0 for i in prob1]
    prob2[index] = 1

    temps = []
    for i in range(len(subs)):
      temp1 = subs[i].split(',')[0]+ ','+subs[i].split(',')[1]+',' +str(prob2[i])
      temps.append(temp1)
    all_content.extend(temps)

  else:
    all_content.extend(subs)

with open('new.csv','w') as f1:
  for row in all_content:
    f1.writelines(row+'\n')