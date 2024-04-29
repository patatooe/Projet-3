import numpy as np
import matplotlib.pyplot as plt
from methode_matrice_2D import methode_matrice_2D_A
from methode_matrice_2D import methode_matrice_2D_b
import imageio.v2
import os
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
import time
import yaml


def methode_matrice_2D_temporelle(planete,  p, l_x, l_z, Lx, Lz, d):

    # Récupération des constantes de la planète
    C_p = planete['C_p']    # Capacité thermique [J/kgK]
    K = planete['K']        # Conductivité thermique [W/mK]
    rho = planete['rho']    # Masse volumique [kg/m^3]
    tau = planete['tau']    # Période de rotation [s]
    Q_0 = planete['Q_0']    # Rayonnement solaire [W/m^2]
    T_s = planete['T_s']    # Température moyenne [K]
    d_pS = planete['d_pS']  # Distance Planète-Soleil [AU]

    alpha = C_p*rho/K # Constante de propagation

    xi = 0.5 # Paramètre xi de la méthode

    # Définition des tailles du domaine
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z

    # Prédéfinition des vecteurs positions
    z = np.linspace(0, Lz, Nz)
    x = np.linspace(0, Lx, Nx)

    # Définition de la distribution de température initiale
    # U0 = np.matrix(np.full((Nx*Nz,1), T_s))

    def initial_gaussian_distribution(Nx, Nz, Lx, Lz, Ts):
        # Création d'une grille Nx par Nz
        x = np.linspace(0, Lx, Nx)
        z = np.linspace(0, Lz, Nz)
        X, Z = np.meshgrid(x, z)

        # Paramètres de la gaussienne centrée
        sigma_x = Lx / 20  # Largeur de la gaussienne en x
        sigma_z = Lz / 20  # Largeur de la gaussienne en z
        mu_x = Lx / 2      # Position x du centre de la gaussienne
        mu_z = Lz / 2      # Position z du centre de la gaussienne

        # Calcul de la distribution gaussienne 2D
        gauss_dist = np.exp(-((X-mu_x)**2 / (2 * sigma_x**2) + (Z-mu_z)**2 / (2 * sigma_z**2)))

        # Ajustement pour le minimum et le maximum
        gauss_dist = Ts + (2 * Ts - Ts) * gauss_dist

        gauss_dist_col = np.ravel(gauss_dist, order='F')

        return gauss_dist_col

    U0 = initial_gaussian_distribution(Nx, Nz, Lx, Lz, T_s)
    U0r = np.reshape(U0,(Nz,Nx),order='F')

    # Impression de la distribution de température initiale
    plt.figure(1)
    plt.pcolor(x, z, U0r, cmap='hot')
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title('T$_0$(x,y) [K]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().invert_yaxis()
    plt.savefig('distribInitiale.png')


    # Définition de la matrice M
    M = eye(Nx*Nz, format='csr')
    M[0, 0] = 0
    M[Nx*Nz-1, Nx*Nz-1] = 0

    # Échelle de temps étudiée
    dt = np.min([(alpha*d**2)/10, tau/20])
    nb_iterations = 62  # Nombre d'itération de temps
    temps_eval = dt*nb_iterations # Temps total d'évaluation

    print(f'Évaluation sur {temps_eval} s avec {nb_iterations} itérations. Pas de {dt} s')

    # Condition de convergence
    if dt>(alpha*d**2) :
        print(f"""
        ##################################################
         ATTENTION! : CRITÈRE DE CONVERGENCE NON RESPECTÉ
            dt = {dt}, alpha*d^2={alpha*d**2}
        ##################################################""")

    # Calculs des matrices initiales et indépendantes du temps
    A = methode_matrice_2D_A(planete, p=p, l_x=l_x, l_z=l_z, Lx=Lx, Lz=Lz, d=d, sparse=True)
    b0 = methode_matrice_2D_b(planete, p=p, l_x=l_x, l_z=l_z, Lx=Lx, Lz=Lz, temps=0, d=d)

    # Construction de A_prime
    A_prime = M - (dt / (alpha * d**2)) * xi * A

    # Initialisation de la boucle avec U0 et b0
    Un = U0
    bn = b0

    # Liste pour enregistrer les images
    images = []

    n=0 # Compteur d'itérations
    for t in np.linspace(dt, temps_eval, nb_iterations+1):
        bn_1 = methode_matrice_2D_b(planete, p=p, l_x=l_x, l_z=l_z, Lx=Lx, Lz=Lz, temps=t, d=d)


        b_prime = (M + dt / (alpha * d**2) * (1 - xi) * A) @ Un - dt / (alpha * d**2) * (xi * bn_1 + (1 - xi) * bn)

        # Solve for Un_1 using sparse solver
        Un_1 = spsolve(A_prime, b_prime)

        # Redéfinition des variables pour la prochaine itération
        Un = np.reshape(Un_1, (len(Un_1), 1))
        bn = np.reshape(bn_1, (len(bn_1), 1))

        # Convertion du vecteur colone de température en matrice dépendant de la position : T_ij->T(x,y)
        Tr = np.reshape(Un, (Nz, Nx), order='F')

        # Eneregistre les images seulement pour certaines itérations
        if n % 1 == 0:
            plt.clf()
            plt.pcolor(x, z, np.array(Tr), cmap='hot', vmin=np.min(np.array(Tr)), vmax=np.max(np.max(np.array(Tr))))
            plt.colorbar(mappable=None, cax=None, ax=None)
            plt.title('T(x,y) [K]')
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            plt.gca().invert_yaxis()
            plt.savefig(f'temperature2d{n}.png')
            images.append(imageio.v2.imread(f'temperature2d{n}.png'))
            os.remove(f'temperature2d{n}.png')
        n += 1

    # Save animation
    imageio.v2.mimsave('temperature2d.gif', images)

with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)

p = 3   # Profondeur de l'abris [m]
l_x = 3 # Largeur de l'abris en x [m]
l_z = 3 # Hauteur de l'abris en z [m]
Lx = 1 # Largeur du domaine [m]
Lz = 3 # Hauteur du domaine [m]
d = 0.01  # Pas de discrétisation [m]


methode_matrice_2D_temporelle(planets_constants['earth'],  p, l_x, l_z, Lx, Lz, d)