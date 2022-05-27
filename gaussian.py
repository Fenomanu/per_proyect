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
    edv = np.zeros(len(alphas))
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

    # Suavizado de parametros y clasificacion
    for i,a in enumerate(alphas):
        print(sigma)
        sigmasuave = a * sigma + (1 - a) * np.identity(D)
        print(sigmasuave)
        G = np.zeros((C,N))
        for c in range(C):
            G[c] = pxc(pc[c], mu[c], sigmasuave[c], Xtr)
        indexes = np.argmax(G, axis=0)
        clasif = xldv[indexes]
        edv[i] = np.mean(xldv!=clasif)*100;
    return edv; # Error de clasificacion del conjunto de validacion


def pxc(pcc, muc, sigmac, X): # Probabilidades de los datos (X) de pertenecer a la clase c
    #print(sigmac)
    pinv = np.linalg.pinv(sigmac)
    Wc = - 0.5 * np.transpose(pinv)
    wc = np.transpose(pinv) @ np.transpose(muc)
    wc01 = np.log(pcc)
    wc02 = 0.5 * logdet(sigmac)
    wc03 = 0.5 * (muc @ np.transpose(pinv)) @ np.transpose(muc)
    #print(str(wc01) + " - " + str(wc02) + " - " + str(wc03))
    wc0 = wc01 - wc02 - wc03
    #print(wc0)
    p1 = np.sum(np.multiply(X @ Wc, X), axis=1)
    pxdc = p1 + (wc @ np.transpose(X)) + wc0
    return pxdc;

def logdet(X): # Para calcular el determinante de la matriz de covarianzas
    # Calculo del logaritmo del determinante (suma de logaritmos de valores propios)
    # Quizas aqui hay que hacer el where==0
    w,v = np.linalg.eig(X)# w = valores, v = vectores
    w = w.real
    if np.any(w <= 0):
        return np.log(sys.float_info.min)
    suma = np.sum(np.log(w))
    print(suma)
    return suma