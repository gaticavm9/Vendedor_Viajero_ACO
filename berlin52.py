import sys
import time
import numpy as np
import pandas as pd
import igraph as ig

sys.argv = ['berlin52.py','1','20','10','0.1','2.5','0.9','berlin52.tsp.txt']

if len(sys.argv) == 8:
    semilla = int(sys.argv[1])
    col = int(sys.argv[2])
    ite = int(sys.argv[3])
    tev = float(sys.argv[4])
    B = float(sys.argv[5])
    q0 = float(sys.argv[6])
    entrada = sys.argv[7]
    print('Parametros de entrada:',semilla, col, ite, tev, B, q0, entrada,'\n')
else:
    print("Error en la entrada de los parámetros")
    print("Los parametros a ingresar son: Semilla, TamañoColonia, NroIteraciones, TasaEvaportacion, Beta, q0, DatosEntrada")
    sys.exit(0)

tiempo_proceso_ini = time.process_time()
np.random.seed(semilla)

##Llenar una matriz con los valores del archivo
matrizCoordenadas = pd.read_table(entrada, header=None, skiprows=6, sep=" ", names=range(3))
matrizCoordenadas = matrizCoordenadas.drop(index=(len(matrizCoordenadas)-1),axis=0)
matrizCoordenadas = matrizCoordenadas.drop(columns=0,axis=1).to_numpy()
numVariables = matrizCoordenadas.shape[0]
print('Matriz de coordenadas:\n', matrizCoordenadas,'\ntamaño',matrizCoordenadas.shape, '\ntipo',type(matrizCoordenadas))
print('Número de variables:', numVariables,'\n')

## Se crea matriz de distancia de 52x52
matrizDistancias=np.full((numVariables,numVariables),fill_value=-1.0,dtype=float)
for i in range(numVariables-1):
    for j in range(i+1,numVariables):
        matrizDistancias[i][j]=np.sqrt(np.sum(np.square(matrizCoordenadas[i]-matrizCoordenadas[j])))
    ##
        matrizDistancias[j][i]=matrizDistancias[i][j]
print('Matriz de Distancias: \n',matrizDistancias,'\ntamaño:',matrizDistancias.shape,'\ntipo:',type(matrizDistancias),'\n')

## Generar matriz heuristica 1/matrizDistancia
matrizHeuristica = np.full_like(matrizDistancias, fill_value=1/matrizDistancias, dtype=float)
print('Matriz de Heurística: \n', matrizHeuristica, '\ntamaño:', matrizHeuristica.shape, '\ntipo:', type(matrizHeuristica),'\n')

#Se procede a crear colonia vacia (tamaño colonia x num variable) inicializado con el valor -1
colonia=np.full((col, numVariables), fill_value=-1, dtype=int)
print('Colonia:\n',colonia, '\ntamaño:', colonia.shape, '\ntipo:', type(colonia),'\n')

#Función para calcular el costo de la solucion
#n: num de var      #s: vector solución     #c: matriz de distancias
def solucionCalculaCosto(n,s,c):
    aux = c[s[n-1]][s[0]]
    for i in range(n-1):
        aux += c[s[i]][s[i+1]]
    return aux

#Creación primera solución
solucionOptima = np.array([0,48,31,44,18,40,7,8,9,42,32,50,10,51,13,12,46,25,26,27,11,24,3,5,14,4,23,47,37,36,39,38,35,34,33,43,45,15,28,49,19,22,29,1,6,41,20,16,2,17,30,21,-2])
solucionMejor = np.arange(0,numVariables)
np.random.shuffle(solucionMejor)
solucionMejorCosto = solucionCalculaCosto(numVariables, solucionMejor, matrizDistancias)
solucionMejorIteracion=0
print('Solucion inicial y a la vez mejor solucion:\n', solucionMejor,'\ntamaño:', solucionMejor.shape,'\ntipo',type(solucionMejor))
print('Costo de la solucion inicial y a la vez mejor solucion: ',solucionMejorCosto)
print('Iteración donde se encontró la mejor solución:', solucionMejorIteracion,'\n')

#Creación Matriz de feromona
matrizFeromona = np.full_like(matrizDistancias,fill_value=1/solucionMejorCosto,dtype=float)
print('Matriz de Feromona: \n',matrizFeromona,'\ntamaño:',matrizFeromona.shape,'\ntipo:',type(matrizFeromona),'\n')
"""
#Inicio ciclo iterativo de ACS por numero predefinido de iteraciones
generacion=0
while generacion < ite:
    generacion+=1
    print('\nGeneracion: ',generacion)

"""
