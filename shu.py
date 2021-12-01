import numpy as np

np.random.seed(10)

numVariables=5
col=3
q0=0.9

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
def proxNodo(nodo):
    FxH=[]
    aux=0
    for i in range(numVariables):
        if(i!=nodo):
            FxH.append(matrizFeromona[nodo][i] * matrizHeuristica[nodo][i])
    print(FxH)
    aux = max(FxH)   
    return aux

    
for i in range(numVariables):  #numVariables
    for j in range(col):  #col
            
        if(np.random.random() <= q0):
            #Formula (1)
            #FxH.append(matrizFeromona[i][j] * matrizHeuristica[i][j])
            #print(proxNodo(colonia[i][j]))
            print(colonia[j][i])
            print("maximo ",proxNodo(colonia[j][i]),"\n")

        #else:
            #Formula (1.2)

    print('aaa','\n')        
    
## np.amax(FxH)



