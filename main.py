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
l_x = l_z = 3 # Largeur de l'abris en x [m] # Hauteur de l'abris en z [m]
Lx = Lz = 10 # Largeur du domaine [m] et  Hauteur du domaine [m]
d = 0.1  # Pas de discrétisation [m]

Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z

z = np.linspace(0, Lz, Nz)
x= np.linspace(0, Lx, Nx)


d_list = [2, 1,  1/2, 1/4, 1/8, 1/16]
t_list = []
mem_list = []
if __name__ == '__main__':
    for i in d_list:
        before_memory = 0
        before_time = time.time_ns()
        methode_matrice_2D_temporelle(planets_constants['earth'], p, l_x, l_z, Lx, Lz, i)
        after_memory = get_memory_usage()
        after_time = time.time_ns()
        peak_time = (after_time-before_time)/1.0e9
        peak_memory = after_memory - before_memory
        t_list.append(peak_time)
        mem_list.append(peak_memory)
        print(f"Peak memory usage: {peak_memory/1024} KB")
        print(f"Time: {peak_time} seconds")
        


poptmem, pcov = curve_fit(fit, (d_list), (mem_list))
a_mem, b_mem = poptmem
mem_list_fit = fit(d_list, a_mem, b_mem)

plt.figure(2)    
plt.loglog((d_list), (mem_list))
plt.loglog((d_list), mem_list_fit, label = "fit lineaire: $t_{{ini}} = {a_mem:.2f})\delta^{{{b_mem:.2f}}}$")
plt.legend()
plt.xlabel('pas de discrétisation')
plt.ylabel('mémoire utilisée (kB)')
plt.title('Mémoire utilisée en fonction du pas de discrétisation')
plt.savefig('memoryEval.png')
plt.show()

popt, pcov = curve_fit(fit, (d_list), (t_list))
a_t, b_t = popt
t_list_fit = fit(d_list, a_t, b_t)


plt.figure(3)
plt.scatter((d_list),(t_list))
plt.loglog((d_list), t_list_fit, label = "fit lineaire: $t_{{ini}} = ({a_t:.2e})\delta^{{{b_t:.2f}}}$")
plt.legend()
plt.xlabel('pas de discrétisation (m)')
plt.ylabel('temps de calcul (s)')
plt.title('Temps de calcul en fonction du pas de discrétisation')
plt.savefig('timeEval.png')
plt.show()



#Appel de nos fonctions
# methode_matrice_2D(C_p, K, rho, tau, Q_0, T_s, d_pS, p, l_x, l_z, temps, N)
# calcul_energie_chauffage()
# calcul_energie_panneaux()
# graph_T()
# graph_E_vs_p()
# graph_E_vs_taille()
# graph_E_vs_ptaille()
