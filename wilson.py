import sys
import math
import numpy as np
from L2dist import L2dist
from scipy import stats
from knn import knn

def wilson(X, xl, k):
    ind = np.array(range(X.shape[0]))
    err = 1
    while (err > 0):
        print(ind.shape)
        err = 0
        V = mnn(X, xl, 100)
        for i in ind: # HACK Probar a hacer el recorrido sobre ind, que tiene más sentido
            i -= err
            c = knnV(V[:,i], ind, xl,k)
            if (xl[i] != c): # HACK Cuidado, que no estamos eliminando elementos de xl
                err += 1
                ind = np.setdiff1d(ind, [i])
    return ind

def mnn(X, xl, m):
  V = np.zeros((m,X.shape[0]))
  # Calculamos la matiz de distancias
  # donde la diagonal representa la distancia de 
  # cada muestra consigo mismo (la cual debe ser infinita)
  #I = np.identity(X.shape[0])
  #np.fill_diagonal(I, float('inf'))
  #D = L2dist(X, X) + I
  for n in range(X.shape[0]):
    YY = np.sum(np.square(X),axis=1);
    xn = X[n,:];
    XX = np.square(xn);
    # XX and YY were converted into row vectors at sum
    # So XX needs to be a column vector and YY is fine as row vector
    D  = XX[:,None] + (YY - 2*xn@np.transpose(X))
    idx = np.argsort(D,axis=0)
    V[:,n] = idx[0,:m] # HACK creo que n y : van al revés
  # Ordenamos de mas cercano a mas lejano cada columna
  # Guardamos en V los primer m vecinos mas cercanos
  # por columnas para cada muestra
  return V

def knnV(Vi, ind, xl,k):
  # Filtramos en idx los indices de Vi
  # que aun no han sido eliminados
  idx = Vi[np.isin(Vi,ind)]
  print(Vi)
  # Escogemos los k primeros
  idx = idx[:k]
  # Realizamos la clasificacion 
  # y asignmamos a c
  classif,_ = stats.mode(xl[idx])
  print(classif)
  c = xl[int(classif[0])]
  return c
