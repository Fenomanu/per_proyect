#!/usr/bin/python3
#./pract01/pca+knn+wilson.py ./MNIST/train-images-idx3-ubyte.npz ./MNIST/train-labels-idx1-ubyte.npz "1 2 5 10 20 50 100 200"

import sys
import math
import numpy as np
from pca import pca
from knn import knn
from wilson import wilson

if len(sys.argv)!=6:
  print('Usage: %s <trdata> <trlabels>  <ks> <%%trper> <%%dvper>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
trper=int(sys.argv[4]);
dvper=int(sys.argv[5]);
ks=np.fromstring(sys.argv[3],dtype=int,sep=' ');

N=X.shape[0];
np.random.seed(23); perm=np.random.permutation(N);
X=X[perm]; xl=xl[perm];

# Selecting a subset for train and dev sets
Ntr=round(trper/100*N);
Xtr=X[:Ntr]; xltr=xl[:Ntr];
Ndv=round(dvper/100*N);
Xdv=X[N-Ndv:]; xldv=xl[N-Ndv:];

X,Y,xl,yl = Xtr,Xdv,xltr,xldv

"""
# Wilson
# Reducimos X
ind = wilson(X,xl,1)
Xw = X[ind]
xlw = xl[ind]

# Llamada al clasificador k-nn (k=1)
# y obtención de la tasa de error
err = knn(Xw, xlw, Y, yl, 1)
print("Error de Wilson sin PCA: " + str(err))
"""

# PCA
m,W = pca(X)

# PCA -> Wilson
# Archivo de salida
file = open("pca-wilson+knn.out", "w")

for k in ks:
  print(str(k))

  # Proyección a k dimensiones
  Wk = W[:,:k]
  Xk = X @ Wk
  Yk = Y @ Wk

  # Reducción de prototipos
  ind = wilson(Xk,xl,1)
  Xkw = Xk[ind]
  xlkw = xl[ind]

  # Llamada al clasificador k-nn (k=1)
  # y obtención de la tasa de error 
  err = knn(Xkw, xlkw, Yk, yl, 1)

  print(str(k) + ": " + str(err))
  # Escritura en .out del resultado
  file.write(str(k)); file.write(' ')
  file.write(str(err)); file.write('\n')

file.close()

"""
# Wilson -> PCA
file = open("wilson-pca+knn.out", "w")

ind = wilson(X,xl,1)
Xw = X[ind]
xlkw = xl[ind]

m,W = pca(Xw)

for k in ks:
  # Proyección a k dimensiones
  Wk = W[:,:k]
  Xwk = Xw @ Wk
  Yk = Y @ Wk

  # Llamada al clasificador k-nn (k=1)
  # y obtención de la tasa de error 
  err = knn(Xwk, xlw, Yk, yl, 1)

  # HACK
  print(str(k))

  # Escritura en .out del resultado
  file.write(str(k)); file.write(' ')
  file.write(str(err)); file.write('\n')

file.close()
"""