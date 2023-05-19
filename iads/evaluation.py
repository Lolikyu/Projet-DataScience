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
#TODO: à compléter  plus tard
# ------------------------ 

#analyse_perfs:
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    n = len(L)
    # moyenne
    somme = 0
    for x in L:
        somme += x
    moyenne = somme / n

    # écart type
    somme_ecart = 0
    for x in L:
        somme_ecart += (x - moyenne)**2
    ecart_type = (somme_ecart / n)**0.5
    return moyenne, ecart_type 



def crossval_strat(X, Y, n_iterations, iteration):
    class_values = np.unique(Y)
    i_test = []
    for v in class_values:
        # indices des exemples de la classe v
        i_v = np.where(Y == v)[0]
        # nombre d'exemples de la classe v
        n_v = len(i_v)
        # indices des exemples de la classe v
        i_test_v = np.arange(int(iteration*n_v/n_iterations), int((iteration+1)*n_v/n_iterations))
        i_test.extend(i_v[i_test_v])
    
    Xtest, Ytest = X[i_test], Y[i_test]
    i_app = np.setdiff1d(np.arange(len(X)), i_test)
    Xapp, Yapp = X[i_app], Y[i_app]
    
    return Xapp, Yapp, Xtest, Ytest



def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
        
    ########################## COMPLETER ICI 
    
    for i in range(nb_iter):
        newC = copy.deepcopy(C)
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        newC.train(Xapp, Yapp)
        perf.append(newC.accuracy(Xtest, Ytest))
    
    ##########################
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)