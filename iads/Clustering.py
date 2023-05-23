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
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import matplotlib.cm as cm

# ------------------------ 

def normalisation(df):
    return (df - df.min()) / (df.max() - df.min())

def dist_euclidienne(v1,v2):
    x1=np.array(v1)
    x2=np.array(v2)
    
    #si le tableau est de dimension 1, on le transforme en tableau de dimension 2
    if len(x1.shape) == 1:
        x1 = x1.reshape((1, -1))
    if len(x2.shape) == 1:
        x2 = x2.reshape((1, -1))
    
    #on remplit les tableaux avec des 0 pour qu'ils aient la même dimension de lignes
    if x1.shape[0] > x2.shape[0]:
        x2 = np.pad(x2, ((0, x1.shape[0]-x2.shape[0]),(0,0)), 'constant')
    else:
        x1 = np.pad(x1, ((0, x2.shape[0]-x1.shape[0]),(0,0)), 'constant')
    
    #on remplit les tableaux avec des 0 pour qu'ils aient la même dimension de colonnes
    if x1.shape[1] > x2.shape[1]:
        x2 = np.pad(x2, ((0,0),(0,x1.shape[1]-x2.shape[1])), 'constant')
    else:
        x1 = np.pad(x1, ((0,0),(0,x2.shape[1]-x1.shape[1])), 'constant')
    
    return np.linalg.norm(x1 - x2)

def centroide(df):
    return (1/len(df))*(np.sum(df, axis=0))

def dist_centroides(d1, d2):
    return dist_euclidienne(centroide(d1), centroide(d2))

def initialise_CHA(df):
    partition = dict()
    for i in range(len(df)):
        partition[i] = [i]
    return partition

def fusionne(df, p0, verbose=False):
    p1 = dict()
    dist_min = float("inf")
    key1 = 0
    key2 = 1
    for i in p0:
            for j in p0:
                if(i < j):
                    dist = dist_centroides(df.iloc[p0[i]], df.iloc[p0[j]])
                    if(dist_min > dist):
                        dist_min = dist
                        key1 = i
                        key2 = j

    for k in p0:
        if(key1 != k and key2 != k):
            p1[k] = p0[k]
    new_key = max(p0.keys()) + 1
    p1[new_key] = p0[key1] + p0[key2]

    if(verbose):
        print("Distance minimale trouvée entre [",key1,", ",key2,"] = ",dist_min)
    return p1, key1, key2, dist_min
    
def CHA_centroid(df):
    liste = []
    p0 = initialise_CHA(df)
    for i in range(len(df)-1):
        p1,k1,k2,dist = fusionne(df,p0)
        liste.append([k1, k2, dist, len(p0[k1])+len(p0[k2])])
        p0 = p1
    return liste

import scipy.cluster.hierarchy
def CHA_centroid(df, verbose=False, dendrogramme=False):
    liste = []
    p0 = initialise_CHA(df)
    for i in range(len(df)-1):
        p1,k1,k2,dist = fusionne(df,p0)
        liste.append([k1, k2, dist, len(p0[k1])+len(p0[k2])])
        p0 = p1
        if(verbose):
            print("Distance minimale trouvée entre [",k1,", ",k2,"] = ",dist)

    if(dendrogramme):
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            CHA_centroid(df), 
            leaf_font_size=24.,
        )
        # Affichage du résultat obtenu:
        plt.show()

    return liste

#Fusion des voisins les plus éloignés
def fusionne_complete(df, p0, verbose=False):
    p1 = dict()
    dist_max = -float("inf")
    key1 = 0
    key2 = 1
    for i in p0:
            for j in p0:
                if(i < j):
                    dist = dist_euclidienne(df.iloc[p0[i]], df.iloc[p0[j]])
                    if(dist_max < dist):
                        dist_max = dist
                        key1 = i
                        key2 = j

    for k in p0:
        if(key1 != k and key2 != k):
            p1[k] = p0[k]
    new_key = max(p0.keys()) + 1
    p1[new_key] = p0[key1] + p0[key2]

    if(verbose):
        print("Distance minimale trouvée entre [",key1,", ",key2,"] = ",dist_max)
    return p1, key1, key2, dist_max

#Fonction qui applique l'algorithme de clustering hierarchique complet
def clustering_hierarchique_complete(df):
    liste = []
    p0 = initialise_CHA(df)
    for i in range(len(df)-1):
        p1,k1,k2,dist = fusionne_complete(df,p0)
        liste.append([k1, k2, dist, len(p0[k1])+len(p0[k2])])
        p0 = p1
    return liste

#Fusion des voisins les plus proches
def fusionne_simple(df, p0, verbose=False):
    p1 = dict()
    dist_min = float("inf")
    key1 = 0
    key2 = 1
    for i in p0:
            for j in p0:
                if(i < j):
                    dist = dist_euclidienne(df.iloc[p0[i]], df.iloc[p0[j]])
                    if(dist_min > dist):
                        dist_min = dist
                        key1 = i
                        key2 = j

    for k in p0:
        if(key1 != k and key2 != k):
            p1[k] = p0[k]
    new_key = max(p0.keys()) + 1
    p1[new_key] = p0[key1] + p0[key2]


    if(verbose):
        print("Distance minimale trouvée entre [",key1,", ",key2,"] = ",dist_min)
    return p1, key1, key2, dist_min
    
#Fonction qui applique l'algorithme de clustering hierarchique simple
def clustering_hierarchique_simple(df):
    liste = []
    p0 = initialise_CHA(df)
    for i in range(len(df)-1):
        p1,k1,k2,dist = fusionne_simple(df,p0)
        liste.append([k1, k2, dist, len(p0[k1])+len(p0[k2])])
        p0 = p1
    return liste

def dist_average(d1,d2):
    return (1/len(d1)*len(d2))*(np.sum(dist_euclidienne(d1,d2)))

#Fusion des voisins dont la distance moyenne est la plus faible
def fusionne_average(df, p0, verbose=False):
    p1 = dict()
    dist_min = float("inf")
    key1 = 0
    key2 = 1
    for i in p0:
            for j in p0:
                if(i < j):
                    dist = dist_average(df.iloc[p0[i]], df.iloc[p0[j]])
                    if(dist_min > dist):
                        dist_min = dist
                        key1 = i
                        key2 = j

    for k in p0:
        if(key1 != k and key2 != k):
            p1[k] = p0[k]
    new_key = max(p0.keys()) + 1
    p1[new_key] = p0[key1] + p0[key2]

    if(verbose):
        print("Distance minimale trouvée entre [",key1,", ",key2,"] = ",dist_min)
    return p1, key1, key2, dist_min
    
#Fonction qui applique l'algorithme de clustering hierarchique simple
def clustering_hierarchique_average(df):
    liste = []
    p0 = initialise_CHA(df)
    for i in range(len(df)-1):
        p1,k1,k2,dist = fusionne_average(df,p0)
        liste.append([k1, k2, dist, len(p0[k1])+len(p0[k2])])
        p0 = p1
    return liste

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  
    df : dataframe
    linkage : type de linkage
    verbise : affichage des distances
    dendrogramme : affichage du dendrogramme
    
    retourne la liste contenat les couples de clusters fusionnés, la distance entre ces deux clusters et la taille du nouveau cluster
    """
    
    liste = []
    p0 = initialise_CHA(DF)
    for i in range(len(DF)-1):
        if linkage == 'complete':
            p1,k1,k2,dist = fusionne_complete(DF,p0)
        elif linkage == 'simple':
            p1,k1,k2,dist = fusionne_simple(DF,p0)
        elif linkage == 'average':
            p1,k1,k2,dist = fusionne_average(DF,p0)
        else:
            p1,k1,k2,dist = fusionne(DF,p0)
            
        liste.append([k1, k2, dist, len(p0[k1])+len(p0[k2])])
        p0 = p1
        if(verbose):
            print("Distance minimale trouvée entre [",k1,", ",k2,"] = ",dist)

    if(dendrogramme):
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        if linkage == 'complete':
            scipy.cluster.hierarchy.dendrogram(
                clustering_hierarchique_complete(DF), 
                leaf_font_size=24.,
            )
        elif linkage == 'simple':
            scipy.cluster.hierarchy.dendrogram(
                clustering_hierarchique_simple(DF), 
                leaf_font_size=24.,
            )
        elif linkage == 'average':
            scipy.cluster.hierarchy.dendrogram(
                clustering_hierarchique_average(DF), 
                leaf_font_size=24.,
            )
        else:
            scipy.cluster.hierarchy.dendrogram(
                CHA_centroid(DF), 
                leaf_font_size=24.,
            )
        # Affichage du résultat obtenu:
        plt.show()

    return liste

def inertie_cluster(ens):
    """ Renvoie l'inertie d'un cluster
    """
    ens = np.array(ens)
    c = centroide(ens)
    return np.sum(dist_euclidienne(ens,c)**2)

import random

def init_kmeans(K,Ens):
    Ens = np.array(Ens)
    indice = [i for i in range(0, len(Ens))]
    rand = random.sample(indice,K)
    return np.array(Ens[rand])

def plus_proche(Exe,Centres):
    indice_centroide = -1
    dist_min = float('inf')
    for i in range(len(Centres)):
        dist = dist_euclidienne(Exe, Centres[i])
        if(dist_min>dist):
            dist_min = dist
            indice_centroide = i
    return indice_centroide

def affecte_cluster(Base,Centres):
    Base = np.array(Base)
    U = dict()
    for i in range(len(Base)):
        key = plus_proche(Base[i],Centres)
        if key in U:
            U[key] = U[key]+[i]
        else:
            U[key] = [i]
    return dict(sorted(U.items()))

def nouveaux_centroides(Base,U):
    Base = np.array(Base)
    centroides = []
    for key,value in U.items():
        centroides.append((np.mean(Base[value], axis=0)))
    return np.array(centroides)

def inertie_globale(Base, U):
    Base = np.array(Base)
    globale = 0
    for key,value in U.items():
        globale += inertie_cluster(Base[value])
    return globale

def kmoyennes(K, Base, epsilon, iter_max):
    Base = np.array(Base)
    Centres = init_kmeans(K,Base)
    U1 = affecte_cluster(Base, Centres)
    for i in range(iter_max):
        Centres = init_kmeans(K,Base)
        U2 = affecte_cluster(Base, Centres)
        if(abs(inertie_globale(Base,U2) - inertie_globale(Base,U1)) < epsilon):
            return nouveaux_centroides(Base,U1), U1
        U1 = U2
    return [], U1

def affiche_resultat(Base,Centres,Affect):
    couleurs = cm.tab20(np.linspace(0, 1, 20))
    Base = np.array(Base)
    if(Centres != []):
        plt.scatter(Centres[:,0],Centres[:,1],color=couleurs[0],marker='x')
        couleurs = np.delete(couleurs,0,0)
    for ((cluster, val), c) in zip(Affect.items(), couleurs):
        data = Base[val]
        plt.scatter(data[:,0],data[:,1],color=c) 
        
#INDEX DE DUNN -> plus l'indice est grande, plus les cluster sont séparés
def Dunn(Base,U):
    Base = np.array(Base)
    dist_min_centroide = float('inf')
    dist_max = -float('inf')
    dist_max_point = []
    
    Ubis = U.copy()
    for c1,v1 in U.items():
        Ubis.pop(c1)
        for c2,v2 in Ubis.items():
            dist_min_centroide = min(dist_min_centroide, dist_centroides(Base[v1],Base[v2]))
        Ubis = U.copy()
    
    for key,val in U.items():
        if(len(val) == 1):
            dist_max_point.append(np.linalg.norm(Base[val]))
        for i in range(len(val)-1):
            dist_max = max(dist_max, dist_euclidienne(Base[i], Base[i+1]))
        dist_max_point.append(dist_max)
    return dist_min_centroide/max(dist_max_point)
    
#INDEX DE XIE-BENI -> plus l'indice est petit, plus le cluster sont sépareés et les points du cluster sont proches

def XieBeni(Base, U):
    Base = np.array(Base)
    sum_cl = 0
    sum_point = 0

    for key,val in U.items():
        centre = centroide(Base[val])
        for i in range(len(val)):
            sum_point += dist_euclidienne(Base[i], centre)**2

    return (1/len(Base))*sum_point