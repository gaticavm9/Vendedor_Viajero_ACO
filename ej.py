import numpy as np

np.random.seed(10)

numVariables=5
col=3
q0=0.9
B=1

matrizHeuristica=np.append([[-1,2,0,4,5]], [[2,-1,8,7,6], [3,8,-1,9,10], [4,7,9,-1,11], [5,6,10,11,-1]], axis=0)

print("\n",matrizHeuristica)
print("\n",matrizHeuristica[0])

for i in range(numVariables):
    if not i in matrizHeuristica[0]:
        print("No está el ", i)
