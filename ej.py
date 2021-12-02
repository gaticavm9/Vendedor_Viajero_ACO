import numpy as np
import random 

np.random.seed(10)

numVariables=5
col=3
q0=0.9
B=1

matrizHeuristica=np.append([[-1,2,0,4,5]], [[2,-1,8,7,6], [3,8,-1,9,10], [4,7,9,-1,11], [5,6,10,11,-1]], axis=0)
prob = [0.3, 0.1, 0.2, 0.2]
posProb = [0, 2, 3, 4] 

"""
print("\n",matrizHeuristica)
print("\n",matrizHeuristica[0])

for i in range(numVariables):
    if not i in matrizHeuristica[0]:
        print("No está el ", i)
"""
selec=random.choices(prob, weights=(prob), k=1)[0]
print(selec)

posAux = prob.index(selec)
pos2 = posProb[posAux]
print("Posicion ",pos2)

sMej = matrizHeuristica[0]
indexI = np.where(sMej == 5)
indexI = int(indexI[0])
print("index ",indexI)
