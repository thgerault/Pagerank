#! /usr/bin/env python3

from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import defaultdict
import pandas as pd
import numpy as np
import time

class PageRank(MRJob):
    """
    On commence par déterminer le nombre de page N
    """

    df=pd.read_table("pagerank.txt")
    page=list(set(df["0"]))
    page_citer=list(set(df["4"]))
    l=list(np.copy(page_citer))
    for p in page:
        if p not in page_citer:
            l.append(p)
    
    N=len(l)

    """
    Ensuite on commence le calcul du pagerank
    """

    def mapper(self, _, line):
        line_split=line.split()
        yield f"lien:Pcite-P {line_split[1]} {line_split[0]}",1  # lien:Pcite-P page cité - page qui cite
        yield f"Ni: {line_split[0]}", 1 # page de gauche

    """
    "lien:Pcite-P 2289 10145"	1
    "Ni: 10145"	1
    
    "lien:Pcite-P 17491 5877"	1
    "Ni: 5877"	1
    """
                                     # key = page citer
    def reducer(self, key, value):   # value = page qui cite la key  
        """
        Pour l'initialisation des poids
        """
        newKey=key.split()
        if newKey[0]=="lien:Pcite-P":
            yield newKey[1], (newKey[2],1/PageRank.N)   # page qui cite, [page citer, w0]
        if newKey[0]=="Ni:":
            yield newKey[1], ("n",sum(value)) # juste somme pour avoir le nb de page que cite chaque page
                                              # page qui cite, [n, nombre de page qu'elle cite]
    """
    "10145"	["2289",1.3178876896110913e-5]
    "17491"	["5877",1.3178876896110913e-5]
    .
    .
    .
    "125"	["n",68]
    "1250"	["n",171]
    """

                                    # key est la page qui cite
    def reducer2(self, key, value): # value = [1, page citer, w0] soit [n, nombre de page citer par key]
        liste=[]
        poids=0
        for v in value:                     # pour chaque element dans les listes rendu par le yield précédent:
            if v[0]=="n":                   # Si on est sur une liste indiquant le ni de la page comme cle
                poids=v[1]                  # on prend le ni
            else: 
                liste+=[(v[0], [key,v[1]])] # sinon on stocke (page citer, [page qui cite, w0])
        
        for element in liste :                             
            if poids != 0:                                 # pour éviter les division par 0 dans la fonction qui suit
                yield element[0], [poids]+element[1]       # page citer, [152] + ["4502",1.3178876896110913e-5] = [ni, page qui cite, w0]
                yield element[0], ('weight', 1/PageRank.N) # page citer, ['weight', w0]

    """
    "27856"	[152,"4502",1.3178876896110913e-5]
    "27856"	["weight",1.3178876896110913e-5]

    "27912"	[152,"4502",1.3178876896110913e-5]
    "27912"	["weight",1.3178876896110913e-5]
    """

    # 1er itération, attribution des w1:

                                     # key = page citer 
    def reducer3(self, key, value):  # value = [ni, page qui cite, w0] soit ['weight', w0]
        somme_poids=0
        weight=0
        for v in value:
            if v[0]=='weight':
                weight=v[1] # w0 de page citer
            else:
                yield v[1], ("est citer par",v[0], key)  # page qui cite, [n de la page qui cite, page citer] 
                somme_poids=somme_poids+(v[2]/v[0])      # somme de tout les wj/nj de la page i

        yield key, ("weight", 0.15*weight+0.85*somme_poids) # page citer, w1


    """
    On a du coup un detail des pages qui sont citer avant d'avoir une ligne avec 
    le premier poids w1. On a :

    ...
    "4494"	["est citer par",5,"4282"]
    "4500"	["est citer par",48,"4282"]
    "4501"	["est citer par",69,"4282"]
    "4502"	["est citer par",152,"4282"]
    "4282"	["weight",0.0006900800405540598]
    """

    # itération suivante qui bouclera
                                            # key = page qui cite soit page citer
    def reducer4(self, key, value):         # value = [n de la page qui cite, page citer] soit [w1 de la key]
        liste=[]
        poids=0

        for v in value:         
            if v[0]=="est citer par" :
                liste+=[(v[2], (v[1],key))] # (page citer, (n, page qui cite))
            
            else :
                poids=v[1]    # stock le poids de la page citer
                yield key, v  # page citer, ["weight",wk] c'est la sauvegarde du poids en quelque sorte
        
        for element in liste :
            liste_n=list(element[1])
            liste_n+=[poids]        # [n] + ["1260",3.1229090973801696e-6]
            yield element[0], liste_n 


    """
    "1260"	["weight",3.1229090973801696e-6]
    "783"	[6,"1260",3.1229090973801696e-6]
    "684"	[6,"1260",3.1229090973801696e-6]
    "550"	[6,"1260",3.1229090973801696e-6]
    "418"	[6,"1260",3.1229090973801696e-6]
    "18"	[6,"1260",3.1229090973801696e-6]
    "12"	[6,"1260",3.1229090973801696e-6]
    """

    def reducer5(self, key, value): 
        """
        Consiste à ne yield que les éléments qui nous interessent, 
        c'est à dire la page et son poids limite wk au bout d'une dizaine d'itération
        """
        for v in value:
            if v[0]=="est citer par" :
                pass
            else :
                yield None, (v[1],key)


    def sort(self,_,value):
        """
        Enfin, cette derniere fonction permet de trier dans l'ordre crois
        """
        liste=list(value)
        liste.sort(reverse=True)
        for pagerank in liste[:10]:
            yield pagerank[1],pagerank[0]

    
    def steps(self):
        return [ MRStep(mapper=self.mapper),MRStep(reducer=self.reducer),MRStep(reducer=self.reducer2) ] + \
            [MRStep(reducer=self.reducer3), MRStep(reducer=self.reducer4)] *10+ \
            [MRStep(reducer=self.reducer3)] + \
            [MRStep(reducer=self.reducer5),MRStep(reducer=self.sort)]
    

if __name__ == '__main__':
    start = time.time()

    PageRank.run()

    end=time.time()
    print("\nTemps total d'exécution : ",end-start, "secondes") # Environ 2 minutes

    """
    Après comparaison avec les résultats de mes camarades, les valeurs trouver 
    ne sont pas les bonnes. Les résultats obtenu ont été stocké dans un fichier texte
    """