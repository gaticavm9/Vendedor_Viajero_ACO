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
    tev = int(sys.argv[4])
    B = int(sys.argv[5])
    q0 = int(sys.argv[6])
    entrada = int(sys.argv[7])
    print(semilla, col, ite, tev, B, q0, entrada)
else:
    print("Error en la entrada de los parámetros")
    print("Los parametros a ingresar son: Semilla, TamañoColonia, NroIteraciones, TasaEvaportacion, Beta, q0, DatosEntrada")
    sys.exit(0)

tiempo_proceso_ini = time.process_time()
np.random.seed(semilla)
matrizCoordenadas = pd.read_table(entrada, header=None, delim_whitespace=True)
matrizCoordenadas = matrizCoordenadas.drop(columns=0,axis=1).to_numpy()
numVariables = matrizCoordenadas.shape[0]
print('Matriz de coordenadas:\n', matrizCoordenadas,'\ntamaño',matrizCoordenadas.shape, '\ntipo',type(matrizCoordenadas))
print('Número de variables:', numVariables)

matrizDistancias=np.full((numVariables,numVariables),fill_value=-1.0,dtype=float)
for i in range(numVariables-1):
    for j in range(i+1,numVariables):
        matrizDistancias[i][j]=np.sqrt(np.sum(np.square(matrizCoordenadas[i]-matrizCoordenadas[j])))


