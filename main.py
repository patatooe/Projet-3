# imports de nos fonctions (from NOM_DU_FICHIER import NOM_DE_LA_FONCTION)
from methode_matrice_2D import methode_matrice_2D
from calcul_energie_chauffage import calcul_energie_chauffage
from calcul_energie_panneaux import calcul_energie_panneaux
from graph_T import graph_T
from graph_E_vs_p import graph_E_vs_p
from graph_E_vs_taille import graph_E_vs_taille
from graph_E_vs_ptaille import graph_E_vs_ptaille

#import des fonctions externes
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import imageio.v2
import os
import time

#Constantes
with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)

p = 3   # Profondeur de l'abris [m]
l_x = 3 # Largeur de l'abris en x [m]
l_z = 3 # Hauteur de l'abris en z [m]
L_x = 5 # Largeur du domaine [m]
L_z = 10 # Hauteur du domaine [m]
d = 0.1  # Pas de discrétisation [m]




#Appel de nos fonctions
# methode_matrice_2D(C_p, K, rho, tau, Q_0, T_s, d_pS, p, l_x, l_z, temps, N)
# calcul_energie_chauffage()
# calcul_energie_panneaux()
# graph_T()
# graph_E_vs_p()
# graph_E_vs_taille()
# graph_E_vs_ptaille()
