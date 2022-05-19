import numpy as np
from collections import Counter

n_max=20
sep=n_max

with open('/data/data_sort_8000_train.txt','w') as f:
    sequences=[]
    sequence=[]
    sorts=[]
    for i in range(8000):
        p_initial=np.random.randint(1,10,20)
        p_initial=p_initial/p_initial.sum()
        p_final=np.random.randint(1,10,20)
        p_final=p_final/p_final.sum()

        for j in range(8000):
            r=((j+1)/8000)
            p=(1-r)*p_initial+r*p_final
            sequence.append(np.random.choice(n_max,1,p=p)[0])
        sort=[item for items, c in Counter(sequence).most_common() for item in [items]]
        for i in range(n_max):
            if i not in sort:
                sort.append(i)
        sequences.append(sequence)
        sorts.append(sort)
        sequence=[]
    for i in range(len(sequences)):
        for j in sequences[i]:
            f.write(str(j)+' ')
        f.write(str(sep)+' ')
        for j in sorts[i]:
            f.write(str(j)+' ')
        f.write('\n')

with open('data/data_sort_8000_valid.txt','w') as f:
    sequences=[]
    sequence=[]
    sorts=[]
    for i in range(800):
        p_initial=np.random.randint(1,10,20)
        p_initial=p_initial/p_initial.sum()
        p_final=np.random.randint(1,10,20)
        p_final=p_final/p_final.sum()

        for j in range(8000):
            r=((j+1)/8000)
            p=(1-r)*p_initial+r*p_final
            sequence.append(np.random.choice(n_max,1,p=p)[0])
        
        sort=[item for items, c in Counter(sequence).most_common() for item in [items]]
        for i in range(n_max):
            if i not in sort:
                sort.append(i)
        sequences.append(sequence)
        sorts.append(sort)
        sequence=[]
    for i in range(len(sequences)):
        for j in sequences[i]:
            f.write(str(j)+' ')
        f.write(str(sep)+' ')
        for j in sorts[i]:
            f.write(str(j)+' ')
        f.write('\n')


with open('data/data_sort_8000_test.txt','w') as f:
    sequences=[]
    sequence=[]
    sorts=[]
    for i in range(800):
        p_initial=np.random.randint(1,10,20)
        p_initial=p_initial/p_initial.sum()
        p_final=np.random.randint(1,10,20)
        p_final=p_final/p_final.sum()

        for j in range(8000):
            r=((j+1)/8000)
            p=(1-r)*p_initial+r*p_final
            sequence.append(np.random.choice(n_max,1,p=p)[0])
        
        sort=[item for items, c in Counter(sequence).most_common() for item in [items]]
        for i in range(n_max):
            if i not in sort:
                sort.append(i)
        sequences.append(sequence)
        sorts.append(sort)
        sequence=[]
    for i in range(len(sequences)):
        for j in sequences[i]:
            f.write(str(j)+' ')
        f.write(str(sep)+' ')
        for j in sorts[i]:
            f.write(str(j)+' ')
        f.write('\n')
