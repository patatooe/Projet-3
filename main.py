# imports de nos fonctions (from NOM_DU_FICHIER import NOM_DE_LA_FONCTION)
from methode_matrice_2D import methode_matrice_2D
from calcul_energie_chauffage import calcul_energie_chauffage
from calcul_energie_panneaux import calcul_energie_panneaux
from graph_T import graph_T

#import des fonctions externes
import numpy as np
import matplotlib.pyplot as plt

#Constantes
C_p = 0
K = 0
rho = 0 
tau = 0 
Q_0 = 0 
T_s = 0 
d_pS = 0 
p = 0
l_x = 0 
l_z = 0
temps = 0 
N = 0


#Appel de nos fonctions
methode_matrice_2D(C_p, K, rho, tau, Q_0, T_s, d_pS, p, l_x, l_z, temps, N)