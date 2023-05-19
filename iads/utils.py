# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
import math
import copy
from iads import evaluation as ev
from iads import Classifiers as cl
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data_desc = np.random.uniform(binf, bsup, (2*n, p))
    data_label = np.array([-1 for i in range(n)] + [+1 for i in range(n)])
    return (data_desc, data_label)
    
def genere_dataset_uniforme(n, p, binf=-1, bsup=1):
    return genere_dataset_uniform(p, n, binf, bsup)

# ------------------------
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    dataset_negative = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    dataset_positive = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    dataset = np.concatenate((dataset_negative, dataset_positive))
    label = np.array([-1 for i in range(nb_points)] + [1 for i in range(nb_points)])
    return (dataset, label)

#-------------------------
def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """
    #base d'apprentissage:
    L_train = random.sample([elem[0] for elem in np.argwhere(label_set == 1)], n_pos) +\
              random.sample([elem[0] for elem in np.argwhere(label_set == -1)], n_neg)
    desc_set_train = desc_set[L_train]
    label_set_train = label_set[L_train]

    #base de test:
    L_test = [i for i in range(0, len(desc_set)) if i not in L_train]
    desc_set_test = desc_set[L_test]
    label_set_test = label_set[L_test]
    
    return (desc_set_train, label_set_train), (desc_set_test, label_set_test)

# ------------------------
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]
    # 'x' rose clair pour la classe -1:
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='x', color="lightpink")
    # '+' cyan foncé pour la classe +1:
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='+', color="darkcyan")

# ------------------------
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["lightgrey","darkgrey"],levels=[-1000,0,1000])

# ------------------------
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    var = np.array([[var, 0],[0, var]])
    data1 = np.random.multivariate_normal(np.array([-0.2,-1]), var, n)
    data2 = np.random.multivariate_normal(np.array([1,1]), var, n)
    data3 = np.random.multivariate_normal(np.array([1,-1]), var, n)
    data4 = np.random.multivariate_normal(np.array([-0.2,1]), var, n)
    
    data_xor = np.concatenate((data1, data2, data3, data4))
    label_xor = np.array([-1 for i in range(2*n)]+[1 for i in range(2*n)])
    
    return data_xor, label_xor

# ------------------------
def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
        
    newC = copy.deepcopy(C)
    for i in range(nb_iter):
        X_train, Y_train, X_test, Y_test = ev.crossval_strat(X, Y, nb_iter, i)
        newC.train(X_train, Y_train)
        perf.append(newC.accuracy(X_test, Y_test))    

    (perf_moy, perf_sd) = ev.analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

# ------------------------
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
        rem: la fonction utilise le log dont la base correspond à la taille de P
    """
    res = 0
    b = len(P)
    if(b==1):
        return 0
    for p in P:
        if(p != 0):
            res += p*math.log(p,b)
    return -res

# ------------------------
def entropie(Y):
    """ Y : (array) : array de labels
        rend la valeur de l'entropie de Shannon correspondante
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return shannon(nb_fois/len(Y))

# ------------------------
def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return valeurs[np.argmax(nb_fois)]

# ------------------------
def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """ 
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = cl.NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        for i in range(X.shape[1]):
            entropie_i = 0
            Xi = np.unique(X[:,i])
            for x in Xi:
                entropie_i += len(Y[X[:,i]==x])/len(Y)*entropie(Y[X[:,i]==x])
            if entropie_i < min_entropie:
                min_entropie = entropie_i
                i_best = i
                Xbest_valeurs = np.unique(X[:,i])
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = cl.NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = cl.NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud