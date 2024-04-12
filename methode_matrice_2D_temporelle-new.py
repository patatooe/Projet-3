import numpy as np
import matplotlib.pyplot as plt
from methode_matrice_2D import methode_matrice_2D_A
from methode_matrice_2D import methode_matrice_2D_b
from distributionInitiale import distributionInitiale
from plot_temperature import plot_temperature
import imageio.v2
import os
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, diags, eye
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import yaml
from tqdm import tqdm

def methode_matrice_2D_temporelle (planete, p, l_x, l_z, Lx, Lz, d ):
    # Définition des constantes
    C_p = planete['C_p']
    K = planete['K']
    rho = planete['rho']
    tau = planete['tau']
    Q_0 = planete['Q_0']
    T_s = planete['T_s']
    d_pS = planete['d_pS']

    alpha=C_p*rho/K

    xi =0.5

    # Calcul du nombre de points en x et z
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
    z = np.linspace(0, Lz, Nz)
    x= np.linspace(0, Lx, Nx)

    # Calcul de la distribution initiale
    # U0 = distributionInitiale(Nx, Nz, Lx, Lz, T_s)
    U0 = np.full((Nx*Nz,1), T_s)
    U0r = np.reshape(U0,(Nz,Nx),order='F')
    plot_temperature(x,z,U0r, 'distribInitiale') # Graphique de Distribution initiale

    # Définition des matrices A, M et b0
    abri = True # Est-ce qu'il y a un abri

    A, M = methode_matrice_2D_A(planete, p, l_x, l_z, Lx, Lz, d, abri=abri)
    b0 = methode_matrice_2D_b(planete, p, l_x, l_z, Lx, Lz, temps=0, d=d, abri=abri)

    # Définition du pas de temps
    dt = np.min([alpha*d**2/4, tau/20 ])
    nb_iterations = 320
    temps_deval = dt*nb_iterations

    if dt>(alpha*d**2) or dt>tau/10 :
        print(f"""
        ##################################################
         ATTENTION! : CRITÈRE DE CONVERGENCE NON RESPECTÉ
            dt = {dt}, alpha*d^2={alpha*d**2}, tau = {tau}
        ##################################################""")    

    # Définition de A_prime
    A_prime = M - dt*xi/(alpha*d**2)*A

    # Images
    images = []

    # Définition des paramètres initiales
    bn = b0
    Un = U0

    n=0
    print(f""" #######################################################################################################
    DÉBUT DU CALCUL 
    PARAMÈTRES : 
    - Domaine : {Lx}x{Lz} m 
    - Abri : {abri}, {l_x}x{l_z} m
    - Discrétisation : Pas de {d} m avec {Nx}x{Nz} points
    - Temps: Évaluation sur {temps_deval} avec un pas de {dt} s sur {nb_iterations} itérations
--------------------------------------------------------------------------------------------------------
    Progression :""")

    for t in tqdm(np.arange(dt, temps_deval, dt), total=nb_iterations):
        bn_1 = methode_matrice_2D_b(planete, p, l_x, l_z, Lx, Lz, temps=t, d=d, abri=abri)


        b_prime_1 = (M + dt*(1-xi)/(alpha*d**2)*A).dot(Un).flatten()
        b_prime_2 = (dt/(alpha*d**2)*(xi*bn_1+(1-xi)*bn)).flatten()

        b_prime = b_prime_1 - b_prime_2

        Un_1 = spsolve(A_prime, b_prime)
        Un=Un_1

        # Plot du graphique au temps t
        Unr = np.reshape(Un_1,(Nz,Nx),order='F')
        plt.clf()
        plot_temperature(x,z,Unr, f'temperature2d{n}')
        images.append(imageio.v2.imread(f'temperature2d{n}.png'))
        os.remove(f'temperature2d{n}.png')

        n=n+1

    imageio.v2.mimsave('temperature2d.gif', images)

    print(f""" 
CALCUL TERMINÉ : ANIMATION SAUVEGARDÉE
#######################################################################################################""")
with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)

p = 3   # Profondeur de l'abris [m]
l_x = 3 # Largeur de l'abris en x [m]
l_z = 3 # Hauteur de l'abris en z [m]
Lx = 10 # Largeur du domaine [m]
Lz = 10 # Hauteur du domaine [m]
d = 0.1  # Pas de discrétisation [m]


methode_matrice_2D_temporelle(planets_constants['earth'],  p, l_x, l_z, Lx, Lz, d)