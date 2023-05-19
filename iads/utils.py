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
import sklearn.utils as sk

# ------------------------ 

# genere_dataset_uniform:
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data1_desc = np.random.uniform(binf, bsup, (2*n,p))
    data1_label = np.asarray([-1 for i in range(0,n)] + [1 for i in range(0,n)])
    sk.shuffle(data1_label)
    return (data1_desc, data1_label)

# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    x_neg = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    x_pos = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    y_pos = np.asarray([1 for i in range(0,nb_points)])
    y_neg = np.asarray([-1 for i in range(0,nb_points)])
    
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    sk.shuffle(y)
    
    return x,y

# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    #TODO: A Compléter
    # Extraction des exemples de classe -1:
    data2_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data2_positifs = desc[labels == +1]
    
    # Affichage de l'ensemble des exemples :
    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1

# plot_frontiere:
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
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])


# create_xor:
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    np.random.seed(42)
    # generation des 4 nuages de points avec une variance var
    X1 = np.random.multivariate_normal(mean=[-1,-1], cov=[[var,0],[0,var]], size=n)
    X2 = np.random.multivariate_normal(mean=[1,1], cov=[[var,0],[0,var]], size=n)
    X3 = np.random.multivariate_normal(mean=[1,-1], cov=[[var,0],[0,var]], size=n)
    X4 = np.random.multivariate_normal(mean=[-1,1], cov=[[var,0],[0,var]], size=n)
    
    # concatenation des nuages de points et des etiquettes
    X = np.vstack((X1, X2, X3, X4))
    Y = np.hstack((-np.ones(n*2), np.ones(n*2)))
    
    return X, Y


# crossval:
def crossval(X, Y, n_iterations, iteration):
    indices = np.arange(len(X))
    size = len(X) // n_iterations
    i_test = indices[iteration*size:(iteration+1)*size]
    
    Xtest, Ytest = X[i_test], Y[i_test]
    Xapp, Yapp = np.delete(X, i_test, axis=0), np.delete(Y, i_test)
    return Xapp, Yapp, Xtest, Ytest


#crossval_strat:
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