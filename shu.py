import numpy as np
import random 

np.random.seed(10)

numVariables=5
col=3
q0=0.7
B=1
tev=0.1

matrizHeuristica=np.append([[-1,2,3,4,5]], [[2,-1,8,7,6], [3,8,-1,9,10], [4,7,9,-1,11], [5,6,10,11,-1]], axis=0)
matrizFeromona=np.full((5,5),fill_value=3,dtype=int)
colonia=np.full((col,5),fill_value=-1,dtype=int)

T0= matrizFeromona[0][0]

print("\n",matrizHeuristica)
print("\n",matrizFeromona)
print("\n",colonia)
print("\n Valor T0",T0)

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

#Funcion Actualizar feromona local
def feromL(ii, jj):
    ferL = ((1-tev)*(matrizFeromona[ii][jj])) + tev*T0
    return ferL

    
for i in range(numVariables):  #numVariables
    print('Valor de i: ', i ,'\n')
    for j in range(col):  #col
        #Hormiga avanza                   
        if(np.random.random() <= q0):
            #Formula (1)
            print(colonia[j][i])
            print("maximo ",proxNodo1(colonia[j][i], j),"\n")
            if(i < numVariables-1):
                colonia[j][i+1] =  proxNodo1(colonia[j][i], j)  #Colocar proximo nodo a visitar

        else:
            if(i < numVariables-1):
                colonia[j][i+1] = proxNodo2(colonia[j][i], j)
        #############
        #Actualizar feromona local
        if(i < numVariables-1):
            matrizFeromona[colonia[j][i]][colonia[j][i+1]] = feromL(colonia[j][i], colonia[j][i+1])


        

    print(colonia, "\n")
    print(matrizFeromona, "\n")        

           
    
## np.amax(FxH)



