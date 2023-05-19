"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Clusters implémentés en LU3IN026
# Version de départ : Février 2023

# Import de packages externes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import scipy.cluster.hierarchy

def normalisation(dataframe):
    min = dataframe.min()
    max = dataframe.max()
    return (dataframe - min) / (max - min)

def dist_euclidienne(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def centroide(df):
    return np.mean(df, axis=0)

def dist_centroides(df1, df2):
    c1 = centroide(df1)
    c2 = centroide(df2)
    return dist_euclidienne(c1, c2)

def fusionne(df, P0, linkage='centroid', verbose=False):
    min_dist = float('inf')
    c1, c2 = None, None
    for i in P0:
        for j in P0:
            if i != j:
                if linkage == 'centroid':
                    dist = dist_centroides(df.iloc[P0[i]], df.iloc[P0[j]])
                elif linkage == 'complete':
                    dist = float('-inf')
                    for x in P0[i]:
                        for y in P0[j]:
                            dist = max(dist, dist_euclidienne(df.iloc[x], df.iloc[y]))
                elif linkage == 'single':
                    dist = float('inf')
                    for x in P0[i]:
                        for y in P0[j]:
                            dist = min(dist, dist_euclidienne(df.iloc[x], df.iloc[y]))
                elif linkage == 'average':
                    dist = 0
                    n = 0
                    for x in P0[i]:
                        for y in P0[j]:
                            dist += dist_euclidienne(df.iloc[x], df.iloc[y])
                            n += 1
                    dist /= n
                if dist < min_dist:
                    min_dist = dist
                    c1, c2 = i, j
    P1 = P0.copy()
    P1[c1] = P1[c1] + P1[c2]
    del P1[c2]
    if verbose:
        print(f'Distance minimale trouvée entre [{c1}, {c2}] = {min_dist}')
    return (P1, c1, c2, min_dist)


def CHA_linkage(df, linkage='centroid', verbose=False, dendrogramme=False):
    n = len(df)
    P0 = {i: [i] for i in range(n)}
    result = []
    while len(P0) > 1:
        P1, c1, c2, min_dist = fusionne(df, P0, linkage=linkage, verbose=verbose)
        result.append([c1, c2, min_dist, len(P1[c1])])
        P0 = P1
    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(
            result,
            leaf_font_size=24.,
        )
        plt.show()
    return result

def CHA_centroid(df, verbose=False, dendrogramme=False):
    n = len(df)
    P0 = {i: [i] for i in range(n)}
    res = []
    while len(P0) > 1:
        P1, c1, c2, min_dist = fusionne(df, P0, verbose=verbose)
        res.append([c1, c2, min_dist, len(P1[c1])])
        P0 = P1
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return res

def CHA(DF,linkage='centroid', verbose=False, dendrogramme=False):
    """  
        input:
            - DF : (dataframe)
            - linkage : (string) définie la méthode de linkage du clustering hiérarchique (centroid par défaut, 
            complete, simple ou average)
            - verbose : (bool) par défaut à False, indique si un message doit être affiché lors de la fusion des 
            clusters en donnant le nom des 2 éléments fusionnés et leur distance
            - dendrogramme : (bool) par défaut à False, indique si on veut afficher le dendrogramme du résultat
    """
    ############################ A COMPLETER
    return CHA_linkage(DF, linkage=linkage, verbose=verbose, dendrogramme=dendrogramme)

