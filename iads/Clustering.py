# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import scipy.cluster.hierarchy as hierarchy
# ------------------------ 
def normalisation(dataframe):
    min = dataframe.min()
    max = dataframe.max()
    return (dataframe - min) / (max - min)

# ------------------------ 
def dist_euclidienne(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# ------------------------ 
def centroide(df):
    return np.mean(df, axis=0)

# ------------------------ 
def dist_centroides(df1, df2):
    c1 = centroide(df1)
    c2 = centroide(df2)
    return dist_euclidienne(c1, c2)

# ------------------------ 
def fusionne(df, P0, linkage='centroid', verbose=False):
    min_dist = float('inf')
    c1, c2 = None, None

    for cluster_id1, cluster_points1 in P0.items():
        for cluster_id2, cluster_points2 in P0.items():
            if cluster_id1 != cluster_id2:
                if linkage == 'centroid':
                    dist = dist_centroides(df.iloc[cluster_points1], df.iloc[cluster_points2])
                elif linkage == 'single':
                    dist = float('inf')
                    for point1 in cluster_points1:
                        for point2 in cluster_points2:
                            dist = min(dist, dist_euclidienne(df.iloc[point1], df.iloc[point2]))
                elif linkage == 'complete':
                    dist = float('-inf')
                    for point1 in cluster_points1:
                        for point2 in cluster_points2:
                            dist = max(dist, dist_euclidienne(df.iloc[point1], df.iloc[point2]))
                elif linkage == 'average':
                    dist = 0
                    n = 0
                    for point1 in cluster_points1:
                        for point2 in cluster_points2:
                            dist += dist_euclidienne(df.iloc[point1], df.iloc[point2])
                            n += 1
                    dist /= n

                if dist < min_dist:
                    min_dist = dist
                    c1, c2 = cluster_id1, cluster_id2

    P1 = P0.copy()
    P1[c1] = P1[c1] + P1[c2]
    del P1[c2]

    if verbose:
        print(f'Distance minimale trouvée entre les clusters [{c1}, {c2}] = {min_dist}')

    return (P1, c1, c2, min_dist)

# ------------------------ 
def CHA_linkage(df, linkage='centroid', verbose=False, dendrogramme=False):
    n = len(df)
    clusters = {i: [i] for i in range(n)}
    result = []
    
    while len(clusters) > 1:
        merged_clusters, c1, c2, min_dist = fusionne(df, clusters, linkage=linkage, verbose=verbose)
        result.append([c1, c2, min_dist, len(merged_clusters[c1])])
        clusters = merged_clusters
    
    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        hierarchy.dendrogram(
            result,
            leaf_font_size=24.,
        )
        plt.show()
    
    return result

# ------------------------ 
def CHA_centroid(df, verbose=False, dendrogramme=False):
    n = len(df)
    clusters = {i: [i] for i in range(n)}
    result = []
    
    while len(clusters) > 1:
        merged_clusters, c1, c2, min_dist = fusionne(df, clusters, linkage='centroid', verbose=verbose)
        result.append([c1, c2, min_dist, len(merged_clusters[c1])])
        clusters = merged_clusters
    
    if dendrogramme:
        fig, ax = plt.subplots(figsize=(30, 15))
        ax.set_title('Dendrogramme', fontsize=25)
        ax.set_xlabel("Indice d'exemple", fontsize=25)
        ax.set_ylabel('Distance', fontsize=25)
        hierarchy.dendrogram(
            result,
            leaf_font_size=24.,
        )
        plt.show()
    
    return result
# ------------------------ 
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
    return CHA_linkage(DF, linkage=linkage, verbose=verbose, dendrogramme=dendrogramme)

# ------------------------ 
