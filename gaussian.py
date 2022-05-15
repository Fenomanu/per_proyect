import numpy as np
import sys
def gaussian(Xtr,xltr,Xdv,xldv,alphas): # Alphas son los valores para el suavizado
    #mu # Matriz de medias por clases
    #sigma # Matriz tridimensional (vector de matrices) con las matrices de covarianza por clases
    # Estimacion parametros por clase
    etqs = np.unique(xltr)
    N = Xtr.shape[0]
    D = Xtr.shape[1]
    C = len(etqs)
    # Calculo de parametros
    pc = np.zeros(C) # Vector de probabilidades a priori de c
    mu = np.zeros((C, D)) # Matriz de mus
    sigma = np.zeros((C, D, D)) # Matriz tridimensional (vector de matrices) con las matrices de covarianza por clases
    # print(N)
    for ind, etq in enumerate(etqs):
        idc = np.where(xltr==etq)[0]
        Nc = len(idc)
        pc[ind] = Nc/N
        Xc = Xtr[idc]
        suma = np.sum(Xc, axis=0)
        mu[ind] = suma/Nc
        op = Xc - mu[ind]
        sigma[ind] = (np.transpose(op) @ op)/Nc
    
    sigma = np.where(sigma==0, sys.float_info.min, sigma)[0]

    # Suavizado de parametros y clasificacion
    for i,a in enumerate(alphas):
        sigmasuave = a * sigma + (1 - a) * np.identity(D)
        G = np.zeros(C)
        for c in range(C):
            G[c] = pxc(pc[c], mu[c], sigma[c], Xtr)
        indexes = np.argmax(G, axis=1)
        clasif = etqs[indexes]
        edv[i] = np.mean(xldv!=clasif)*100;
    return edv; # Error de clasificacion del conjunto de validacion


def pxc(pcc, muc, sigmac, X): # Probabilidades de los datos (X) de pertenecer a la clase c
    pinv = np.linalg.pinv(sigmac)
    Wc = -pinv/2
    wc = - muc @ pinv
    wc0 = np.log(pcc) - logdet(sigmac)/2 - np.transpose(muc) @ pinv @ muc
    pxdc = np.transpose(Xtr) @ Wc @ Xtr + np.transpose(wc) @ Xtr + wc0
    return pxdc;

def logdet(X): # Para calcular el determinante de la matriz de covarianzas
    # Calculo del logaritmo del determinante (suma de logaritmos de valores propios)
    # Quizas aqui hay que hacer el where==0
    w,v = np.eig(X)# w = valores, v = vectores
    return np.sum(np.log(w))