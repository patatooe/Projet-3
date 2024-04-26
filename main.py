# imports de nos fonctions (from NOM_DU_FICHIER import NOM_DE_LA_FONCTION)
from methode_matrice_2D import methode_matrice_2D_A
from methode_matrice_2D import methode_matrice_2D_b
from methode_matrice_2D_temporelle import methode_matrice_2D_temporelle

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
from scipy.optimize import curve_fit
import imageio.v2
import os
import time
from memory_profiler import profile
import psutil

# Define the function to fit
def fit(x, a, b,):
    return a * x**b

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss

#Constantes
with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)



p = 3   # Profondeur de l'abris [m]
l_x = l_z = 1.5 # Largeur de l'abris en x [m] # Hauteur de l'abris en z [m]
Lx = Lz = 8 # Largeur du domaine [m] et  Hauteur du domaine [m]
d = 0.1  # Pas de discrétisation [m]

Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z

z = np.linspace(0, Lz, Nz)
x= np.linspace(0, Lx, Nx)


p_list = np.linspace(0,3, 10)
t_list = []
mem_list = []
energy_p = np.zeros([np.size(p_list)])
i=0
if __name__ == '__main__':
    for p in p_list:
        energy_p[i] =  methode_matrice_2D_temporelle(planets_constants['earth'], p, l_x, l_z, Lx, Lz, d)
        i+=1



plt.figure(2)    
plt.plot((p_list), (energy_p))
# plt.loglog((p_list), mem_list_fit, label = "fit lineaire: $t_{{ini}} = {a_mem:.2f})\delta^{{{b_mem:.2f}}}$")
plt.legend()
plt.xlabel('Profondeur')
plt.ylabel('Énergie')
# plt.title('Mémoire utilisée en fonction du pas de discrétisation')
plt.savefig('energy_p.png')
plt.show()
