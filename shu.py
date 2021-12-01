import numpy as np

np.random.seed(10)

numVariables=5
col=3
q0=0.9
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

##Funcion que imprime grafo con las conexiones de las n variables
def proxNodo(nodo, hormiga):
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

    
for i in range(numVariables):  #numVariables
    print('Valor de i: ', i ,'\n')
    for j in range(col):  #col                   
        if(np.random.random() <= q0):
            #Formula (1)
            print(colonia[j][i])
            print("maximo ",proxNodo(colonia[j][i], j),"\n")
            if(i < numVariables-1):
                colonia[j][i+1] =  proxNodo(colonia[j][i], j)  #Colocar proximo nodo a visitar
        #else:
            #Formula (1.2)
    print(colonia, "\n")
            

           
    
## np.amax(FxH)



