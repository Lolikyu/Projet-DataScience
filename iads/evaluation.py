# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
def crossval_strat(X, Y, n_iterations, iteration):
    c = Y[0]
    li_1 = np.where(Y == c)
    li_2 = np.where(Y != c)
    X1 = X[li_2]
    X2 = X[li_1]
    Y1 = Y[li_2]
    Y2 = Y[li_1]
    Xtest = np.concatenate((X1[iteration*(len(X1)//n_iterations):(iteration+1)*(len(X1)//n_iterations)], X2[iteration*(len(X2)//n_iterations):(iteration+1)*(len(X2)//n_iterations)]))
    Ytest = np.concatenate((Y1[iteration*(len(Y1)//n_iterations):(iteration+1)*(len(Y1)//n_iterations)], Y2[iteration*(len(Y2)//n_iterations):(iteration+1)*(len(Y2)//n_iterations)]))
    L1 = li_2[0][iteration*(len(X1)//n_iterations):(iteration+1)*(len(X1)//n_iterations)]
    L2 = li_1[0][iteration*(len(X2)//n_iterations):(iteration+1)*(len(X2)//n_iterations)]
    L = np.concatenate((L1, L2))
    Xapp = np.delete(X, L,0)
    Yapp = np.delete(Y, L,0)   
    return Xapp, Yapp, Xtest, Ytest

# ------------------------ 
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = sum(L)/len(L)
    ecart_type = np.sqrt(sum([(x-moyenne)**2 for x in L])/len(L))
    return (moyenne, ecart_type)

# ------------------------ 
def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
        
    newC = copy.deepcopy(C)
    for i in range(nb_iter):
        X_train, Y_train, X_test, Y_test = crossval_strat(X, Y, nb_iter, i)
        newC.train(X_train, Y_train)
        perf.append(newC.accuracy(X_test, Y_test))    

    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)