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
import math

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

    xi =1

    # Calcul du nombre de points en x et z
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
    z = np.linspace(0, Lz, Nz)
    x= np.linspace(0, Lx, Nx)

    # Calcul de la distribution initiale
    U0 = distributionInitiale(Nx, Nz, Lx, Lz, T_s, l_x, l_z, p)
    # U0 = np.full((Nx*Nz,1), T_s)
    U0r = np.reshape(U0,(Nz,Nx),order='F')
    plot_temperature(x,z,U0r, 'distribInitiale') # Graphique de Distribution initiale
    
    # Définition des matrices A, M et b0
    abri = True # Est-ce qu'il y a un abri

    A, M = methode_matrice_2D_A(planete, p, l_x, l_z, Lx, Lz, d, abri=abri)
    b0 = methode_matrice_2D_b(planete, p, l_x, l_z, Lx, Lz, temps=0, d=d, abri=abri)
    
    # U0m = spsolve(A.tocsr(), methode_matrice_2D_b(planete, p, l_x, l_z, Lx, Lz, temps=46000, d=d, abri=abri)) 
    # plot_temperature(x,z,U0m, 'distribInitiale')

    # Définition du pas de temps
    dt = np.min([alpha*d**2/2, tau/5 ])
    nb_iterations = 100
    temps_deval = dt*nb_iterations
    temps_deval = 2*tau
    dt = temps_deval/nb_iterations

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
    Energy = [] #Liste pour stocker les energie a chaque temps t
    for t in tqdm(np.arange(dt, temps_deval, dt), total=nb_iterations):
        bn_1 = methode_matrice_2D_b(planete, p, l_x, l_z, Lx, Lz, temps=t, d=d, abri=abri)


        b_prime_1 = (M + dt*(1-xi)/(alpha*d**2)*A).dot(Un).flatten()
        b_prime_2 = (dt/(alpha*d**2)*(xi*bn_1+(1-xi)*bn)).flatten()

        b_prime = b_prime_1 - b_prime_2
        
        Un_1 = spsolve(A_prime, b_prime)
        Un=Un_1


        # Plot du graphique au temps t
        Unr = np.reshape(Un_1,(Nz,Nx),order='F')
        plot_temperature(x,z,Unr, f'temperature2d{n}')
        images.append(imageio.v2.imread(f'temperature2d{n}.png'))
        os.remove(f'temperature2d{n}.png')

        if t>tau:
            #ENERGIE CODE*****************************************************************************
            P_tot = 0
            P_updown = 0 #Puissance en W/m
            P_side = 0 #Puissance en W/m
            for i in range(Nz):
                for j in range(Nx):
                    if i == int(p/d) and j <= (l_x/(2*d)):
                        dT_up = (3*Unr[i,j] - 4*Unr[i-1,j] + Unr[i-2,j])/(2*d) #Formule diff finie arriere second degre 
                        P_updown += abs(K*l_x*dT_up)
                    if i == int((l_z+p)/d) and j <= (l_x/(2*d)):
                        dT_down = (-3*Unr[i,j] + 4*Unr[i+1,j] - Unr[i+2,j])/(2*d) #Formule diff finie avant second degre
                        P_updown += abs(K*l_x*dT_down)
                    if (p/d) < i < ((l_z+p)/d) and  j == int(l_x/(2*d)):
                        dT_side = (-3*Unr[i,j] + 4*Unr[i,j+1] - Unr[i,j+2])/(2*d) #Formule diff finie avant second degre
                        P_side += abs(K*l_z*dT_side)
            
            P_tot = P_side+P_updown
            Energy.append((P_tot)*(temps_deval-tau))
            #ENERGIE FIN***********************************************************************************
        


        n=n+1

    imageio.v2.mimsave('temperatureEarth.gif', images)
    print("Energie totale (J/m) pour une periode de temps", (temps_deval -tau), "sec = ", (np.sum(np.array(Energy))/1000), "KJ/m")
    print(f""" 
CALCUL TERMINÉ : ANIMATION SAUVEGARDÉE
#######################################################################################################""")
    return np.sum(np.array(Energy))


with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)

p = 1   # Profondeur de l'abris [m]
l_x = 1 # Largeur de l'abris en x [m]
l_z = 1 # Hauteur de l'abris en z [m]
Lx = 4 # Largeur du domaine [m]
Lz = 4 # Hauteur du domaine [m]
d = 0.05  # Pas de discrétisation [m]


methode_matrice_2D_temporelle(planets_constants['earth'],  p, l_x, l_z, Lx, Lz, d)

# profondeur = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5]
# Energy_requise = []
# for p in profondeur:
#     Energy_requise.append(methode_matrice_2D_temporelle(planets_constants['earth'],  p, l_x, l_z, Lx, Lz, d))
    
# plt.plot(profondeur, Energy_requise)
# plt.xlabel("Profondeur de l'abri (m)")
# plt.ylabel("Energy (KJ/m)")
# plt.show()