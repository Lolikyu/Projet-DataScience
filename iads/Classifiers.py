# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2023

# Import de packages externes
import numpy as np
import pandas as pd
import random

# ---------------------------


# Recopier ici la classe Classifier (complète) du TME 2
class Classifier:
	""" Classe (abstraite) pour représenter un classifieur
		Attention: cette classe est ne doit pas être instanciée.
	"""
    
	def __init__(self, input_dimension):
		""" Constructeur de Classifier
			Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
		self.dimension = input_dimension
        
	def train(self, desc_set, label_set):
		""" Permet d'entrainer le modele sur l'ensemble donné
			desc_set: ndarray avec des descriptions
			label_set: ndarray avec les labels correspondants
			Hypothèse: desc_set et label_set ont le même nombre de lignes
		"""        
		raise NotImplementedError("Please Implement this method")
    
	def score(self,x):
		""" rend le score de prédiction sur x (valeur réelle)
			x: une description
		"""
		raise NotImplementedError("Please Implement this method")
    
	def predict(self, x):
		""" rend la prediction sur x (soit -1 ou soit +1)
			x: une description
		"""
		raise NotImplementedError("Please Implement this method")

	def accuracy(self, desc_set, label_set):
		""" Permet de calculer la qualité du système sur un dataset donné
			desc_set: ndarray avec des descriptions
			label_set: ndarray avec les labels correspondants
			Hypothèse: desc_set et label_set ont le même nombre de lignes
		"""
        
		nb_correct = 0
		for i in range(len(desc_set)):
			prediction = self.predict(desc_set[i])
			if prediction == label_set[i]:
				nb_correct += 1
		return nb_correct / len(desc_set)


class ClassifierKNN(Classifier):
	""" Classe pour représenter un classifieur par K plus proches voisins.
		Cette classe hérite de la classe Classifier
	"""

	# ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
	def __init__(self, input_dimension, k):
		""" Constructeur de Classifier
			Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
		super().__init__(input_dimension)
		self.k = k

	def score(self,x):
		""" rend la proportion de +1 parmi les k ppv de x (valeur réelle)
			x: une description : un ndarray
		"""
		dist = np.linalg.norm(self.desc_set-x, axis=1)
		argsort = np.argsort(dist)[:self.k]
		p = sum(1 for i in argsort if self.label_set[i] == 1) / self.k
		return 2 * (p - 0.5)  

	def predict(self, x):
		""" rend la prediction sur x (-1 ou +1)
			x: une description : un ndarray
		"""
		if self.score(x) < 0:
			return -1
		else:
			return 1

	def train(self, desc_set, label_set):
		""" Permet d'entrainer le modele sur l'ensemble donné
			desc_set: ndarray avec des descriptions
			label_set: ndarray avec les labels correspondants
			Hypothèse: desc_set et label_set ont le même nombre de lignes
		"""        
		self.desc_set = desc_set
		self.label_set = label_set


class ClassifierLineaireRandom(Classifier):
	""" Classe pour représenter un classifieur linéaire aléatoire
		Cette classe hérite de la classe Classifier
	"""
    
	def __init__(self, input_dimension):
		""" Constructeur de Classifier
			Argument:
				- intput_dimension (int) : dimension de la description des exemples
			Hypothèse : input_dimension > 0
		"""
		super().__init__(input_dimension)
		self.w = np.random.uniform(-1,1,input_dimension)
		norm = np.sqrt(np.sum(np.power(self.w,2)))
		self.w = self.w / norm
        
	def train(self, desc_set, label_set):
		""" Permet d'entrainer le modele sur l'ensemble donné
			desc_set: ndarray avec des descriptions
			label_set: ndarray avec les labels correspondants
			Hypothèse: desc_set et label_set ont le même nombre de lignes
		"""        
		print("Pas d'apprentissage pour ce classifieur")
    
	def score(self,x):
		""" rend le score de prédiction sur x (valeur réelle)
			x: une description
		"""
		res = 0
		for i in range(len(x)):
			res += x[i] * self.w[i]
		return res
    
	def predict(self, x):
		""" rend la prediction sur x (soit -1 ou soit +1)
			x: une description
		"""
		if self.score(x) >= 0:
			return 1
		else:
			return -1
            


class ClassifierPerceptron(Classifier):
	""" Perceptron de Rosenblatt
	"""
	def __init__(self, input_dimension, learning_rate=0.01, init=True ):
		""" Constructeur de Classifier
			Argument:
				- input_dimension (int) : dimension de la description des exemples (>0)
				- learning_rate (par défaut 0.01): epsilon
				- init est le mode d'initialisation de w: 
					- si True (par défaut): initialisation à 0 de w,
					- si False : initialisation par tirage aléatoire de valeurs petites
		"""
		super().__init__(input_dimension)
		self.learning_rate = learning_rate
		if init:
			self.w = np.zeros(self.dimension)
		else:
			self.w = np.array([((2*np.random.rand(0,1)-1)*0.001) for i in range(input_dimension)])
		self.allw = [self.w.copy()] # stockage des premiers poids

	def train_step(self, desc_set, label_set):
		""" Réalise une unique itération sur tous les exemples du dataset
			donné en prenant les exemples aléatoirement.
			Arguments:
				- desc_set: ndarray avec des descriptions
				- label_set: ndarray avec les labels correspondants
		"""        
		data_indice = list(range(len(desc_set)))
		np.random.shuffle(data_indice)
		for i in data_indice:
			if self.predict(desc_set[i]) != label_set[i]:
				self.w = self.w + self.learning_rate * label_set[i] * desc_set[i]
				self.allw.append(self.w.copy())
     
	def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
		""" Apprentissage itératif du perceptron sur le dataset donné.
			Arguments:
				- desc_set: ndarray avec des descriptions
				- label_set: ndarray avec les labels correspondants
				- nb_max (par défaut: 100) : nombre d'itérations maximale
				- seuil (par défaut: 0.001) : seuil de convergence
			Retour: la fonction rend une liste
				- liste des valeurs de norme de différences
		"""        
		diff = 0
		diffs = []
		for i in range(nb_max):
			w_previous = np.copy(self.w)
			self.train_step(desc_set, label_set)
			diff = np.sqrt(np.sum((w_previous - self.w)**2))
			diffs.append(diff)
			if diff < seuil:
				break
		return diffs
    
	def score(self,x):
		""" rend le score de prédiction sur x (valeur réelle)
			x: une description
		"""
		return np.dot(x, self.w)

	def predict(self, x):
		""" rend la prediction sur x (soit -1 ou soit +1)
			x: une description
		"""
		if self.score(x) < 0:
			return -1
		else:
			return 1

	def get_allw(self):
		""" getter de allw """
		return self.allw

		
class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        ### A COMPLETER !
        # Ne pas oublier d'ajouter les poids à allw avant de terminer la méthode
        np.random.seed(42)
        indices = np.arange(len(desc_set))
        np.random.shuffle(indices)
        for i in indices:
            if self.score(desc_set[i]) * label_set[i] < 1:
                self.w = self.w + self.learning_rate * (label_set[i] - self.score(desc_set[i])) * desc_set[i]
                self.allw.append(self.w.copy())


#classe majoritaire
def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    #### A compléter pour répondre à la question posée
    unique_labels, counts = np.unique(Y, return_counts=True)
    index_max_count = np.argmax(counts)
    return unique_labels[index_max_count]


def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        for i in range(X.shape[1]):
            entropie_i = 0
            Xi = X[:,i]
            valeurs_i = set(Xi)
            for v in valeurs_i:
                Yi = Y[Xi == v]
                entropie_i += len(Yi)/len(Y) * entropie(Yi)
            if entropie_i < min_entropie:
                min_entropie = entropie_i
                i_best = i
                Xbest_valeurs = valeurs_i
        
        
        
        ############
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


#shannon
import math
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
        rem: la fonction utilise le log dont la base correspond à la taille de P
    """
    #### A compléter pour répondre à la question posée
    if len(P) == 1:
        return 0.0
    H = 0
    for p in P:
        if p == 0:
            continue
        H += p * math.log(p, len(P))
    return -H


#entropie
def entropie(Y):
    unique_labels, counts = np.unique(Y, return_counts=True)
    proportions = counts / len(Y)
    return shannon(proportions)


#Noeud categoriel
class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


#classe arbre decision
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
        
        ##################
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        return self.racine.classifie(x)
        ##################

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)



class ClassifierKNN_MC(Classifier):
    def __init__(self, input_dimension, k, nb_classes):
        super().__init__(input_dimension)
        self.k = k
        self.nb_classes = nb_classes
        
    def score(self,x):
        dist = np.linalg.norm(self.desc_set-x, axis=1)
        argsort = np.argsort(dist)
        scores = np.zeros(self.nb_classes)
        for i in range(self.nb_classes):
            scores[i] = np.sum(self.label_set[argsort[:self.k]] == i)
        return scores / self.k

    def predict(self, x):
        return np.argmax(self.score(x))

    def train(self, desc_set, label_set):
        self.desc_set = desc_set
        self.label_set = label_set