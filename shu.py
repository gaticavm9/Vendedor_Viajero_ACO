import numpy as np
import random

np.random.seed(10)

aa=np.full((10,10),fill_value=1,dtype=int)
bb=np.full((7,10),fill_value=-1,dtype=int)

"""
print(aa)
print("\n",bb)
print("\n",len(aa),len(bb))
for i in range(len(bb)):
    for j in range(len(aa)):
        aa[i][j]=bb[i][j] 
print("\n",aa)

np.random.shuffle(aa)
"""

print("\n",bb)


bb[:, 0] =  np.random.randint(0, 5, size=(1, 7))
print("\n",bb)
