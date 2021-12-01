import numpy as np
import random 

np.random.seed(10)

numVariables=5
col=3
q0=0.5
B=1

matrizHeuristica=np.append([[-1,2,3,4,5]], [[2,-1,8,7,6], [3,8,-1,9,10], [4,7,9,-1,11], [5,6,10,11,-1]], axis=0)
matrizFeromona=np.full((5,5),fill_value=3,dtype=int)
colonia=np.full((col,5),fill_value=-1,dtype=int)


print("\n",matrizHeuristica)
print("\n",matrizFeromona)
print("\n",colonia)


colonia[:, 0] =  np.random.randint(0, numVariables, size=(1, col)) #Llenar
print("\n",colonia)

FxH=[]

##Funcion 1.1 selecciona el nodo siguiente a visitar para el caso 0 < x < q0
def proxNodo1(nodo, hormiga):
    FxH=0
    max=0
    pos=0
    for i in range(numVariables):
        if not i in colonia[hormiga]:  #Restringir que ya está visitado (Solo nodo actual if(i!=nodo):)
            FxH = matrizFeromona[nodo][i] * (matrizHeuristica[nodo][i] ** B)
            if(FxH > max):
                max = FxH
                pos = i  
    return pos

##Funcion 2.2 selecciona el nodo siguiente a visitar para el caso q0 < x < 1
def proxNodo2(nodo, hormiga):
    prob=[]
    posProb=[]
    pos2=0
    sumFxH=0
    #Sumatoria de FxH**B
    for i in range(numVariables):
        if not i in colonia[hormiga]:
            sumFxH= sumFxH + (matrizFeromona[nodo][i] * (matrizHeuristica[nodo][i] ** B))
    #Hallar probabilidades y guardarlas en una lista
    for i in range(numVariables):
        if not i in colonia[hormiga]:  #Restringir que ya está visitado (Solo nodo actual if(i!=nodo):)
            prob.append( (matrizFeromona[nodo][i] * (matrizHeuristica[nodo][i] ** B)) / sumFxH )
            posProb.append(i)
    #Seleccionar un elemento con ruleta
    selec=random.choices(prob, weights=(prob), k=1)[0]
    posAux = prob.index(selec)
    pos2 = posProb[posAux]     

    return pos2


    
for i in range(numVariables):  #numVariables
    print('Valor de i: ', i ,'\n')
    for j in range(col):  #col                   
        if(np.random.random() <= q0):
            #Formula (1)
            print(colonia[j][i])
            print("maximo ",proxNodo1(colonia[j][i], j),"\n")
            if(i < numVariables-1):
                colonia[j][i+1] =  proxNodo1(colonia[j][i], j)  #Colocar proximo nodo a visitar

        else:
            if(i < numVariables-1):
                colonia[j][i+1] = proxNodo2(colonia[j][i], j)

            #Formula (1.2)

    print(colonia, "\n")
            

           
    
## np.amax(FxH)



