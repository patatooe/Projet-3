import numpy as np  
import yaml  
from methode_matrice_2D import methode_matrice_2D_A
from methode_matrice_2D import methode_matrice_2D_b
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from plot_temperature import plot_temperature

#------------------------------------------------------------------------------------- Définition de la distribution Gaussienne centrée
# def distributionInitiale (Nx, Nz, Lx, Lz, Ts):
#         # Création d'une grille Nx par Nz
#         x = np.linspace(0, Lx, Nx)
#         z = np.linspace(0, Lz, Nz)
#         X, Z = np.meshgrid(x, z)

#         # Paramètres de la gaussienne centrée
#         sigma_x = Lx / 20  # Largeur de la gaussienne en x
#         sigma_z = Lz / 20  # Largeur de la gaussienne en z
#         mu_x = Lx / 2      # Position x du centre de la gaussienne
#         mu_z = Lz / 2      # Position z du centre de la gaussienne

#         # Calcul de la distribution gaussienne 2D
#         gauss_dist = np.exp(-((X-mu_x)**2 / (2 * sigma_x**2) + (Z-mu_z)**2 / (2 * sigma_z**2)))

#         # Ajustement pour le minimum et le maximum
#         gauss_dist = Ts + (2 * Ts - Ts) * gauss_dist

#         gauss_dist_col = np.ravel(gauss_dist, order='F')

#         return gauss_dist_col


#------------------------------------------------------------------------------- Définition de la distribution initiale avec abri
'''
def distributionInitiale(Nx, Nz, Lx, Lz, Ts, lx, lz, p):
    # Création d'une grille Nx par Nz
    x = np.linspace(0, Lx, Nx)
    z = np.linspace(0, Lz, Nz)
    X, Z = np.meshgrid(x, z)

    # Initialisation de la distribution de température
    init_dist = np.zeros((Nz, Nx))

    # Parcours de chaque point de la grille
    for i in range(Nz):
        for j in range(Nx):
            if X[i, j] <= lx/2 and (p + lz) >= Z[i, j] >= p:
                init_dist[i, j] = 294.5  # Inside the domain
            else:
                init_dist[i, j] = Ts  # Outside the domain

    init_dist_col = np.ravel(init_dist, order='F')

    return init_dist_col
'''

#--------------------------------------------------------------------------- Définition de la distribution initiale independante du temps
def distributionInitiale(planete, Nx, Nz, Lx, Lz, lx, lz, p, d, temps, abri):
    # Création d'une grille Nx par Nz
    x = np.linspace(0, Lx, Nx)
    z = np.linspace(0, Lz, Nz)
    X, Z = np.meshgrid(x, z)

    with open('constants.yaml') as f:
        planets_constants = yaml.safe_load(f)

    A, M = methode_matrice_2D_A(planets_constants[planete], p=p, l_x=lx, l_z=lz, Lx=Lx, Lz=Lz, d=d, abri=abri)
    b = methode_matrice_2D_b(planets_constants[planete], p=p, l_x=lx, l_z=lz, Lx=Lx, Lz=Lz, temps = temps, d=d, abri=abri)
    
    # Résolution du système d'équations
    T=np.zeros((Nx*Nz,1),dtype=np.double)
    Tr=np.zeros((Nz,Nx),dtype=np.double)

    T = spsolve(A.tocsr(), b) # À utiliser is matrice pleine
    Tr=np.reshape(T,(Nz,Nx),order='F') # Convertion du vecteur colone de température en matrice dépendant de la position : T_ij->T(x,y)

    # Initialisation de la distribution de température
    init_dist = np.zeros((Nz, Nx))
    init_dist = Tr

    plot_temperature(x,z,init_dist,f'distributionInitiale_IndepTemp')

    init_dist_col = np.ravel(init_dist, order='F')

    return init_dist_col

#----------------------------------------------------------------------------------- Affichage des données
'''
planete = 'earth'
p = 1   # Profondeur de l'abris [m]
lx = 1 # Largeur de l'abris en x [m]
lz = 1 # Hauteur de l'abris en z [m]
Lx = 3 # Largeur du domaine [m]
Lz = 3 # Hauteur du domaine [m]
d = 0.05  # Pas de discrétisation [m]
Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
x = np.linspace(0, Lx, Nx)
z = np.linspace(0, Lz, Nz)
temps=0
abri=True

init_Col = distributionInitiale(planete, Nx, Nz, Lx, Lz, lx, lz, p, d, temps, abri)

'''