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
    Xtest = X[iteration*(len(X)//n_iterations):(iteration+1)*(len(X)//n_iterations)]
    Ytest = Y[iteration*(len(Y)//n_iterations):(iteration+1)*(len(Y)//n_iterations)]
    Xapp = np.concatenate((X[0:iteration*(len(X)//n_iterations)], X[(iteration+1)*(len(X)//n_iterations):len(X)]))
    Yapp = np.concatenate((Y[0:iteration*(len(Y)//n_iterations)], Y[(iteration+1)*(len(Y)//n_iterations):len(Y)]))   
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