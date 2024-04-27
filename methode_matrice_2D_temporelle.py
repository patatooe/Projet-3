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

    # Définition du pas de temps
    dt = np.min([alpha*d**2/2, tau/5 ]) #1875 ou 2100000
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
    Energy = []
    for t in tqdm(np.arange(dt, temps_deval, dt), total=nb_iterations):
        bn_1 = methode_matrice_2D_b(planete, p, l_x, l_z, Lx, Lz, temps=t, d=d, abri=abri)


        b_prime_1 = (M + dt*(1-xi)/(alpha*d**2)*A).dot(Un).flatten()
        b_prime_2 = (dt/(alpha*d**2)*(xi*bn_1+(1-xi)*bn)).flatten()

        b_prime = b_prime_1 - b_prime_2

        def Aindex(i,j): #Associé la case i,j à sa colone dans la matrice M
            index=(j-1)*Nz+i
            return index-1

        


        Un_1 = spsolve(A_prime, b_prime)
        # for i in np.arange(1,Nz+1,1): #i=1,..,Nz - numérotation des nœuds sur un maillage physique
        #     for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
        #         if i > p/d+1  and i < (l_z+p)/d+1 and j < l_x/(2*d)+1 and abri:
        #             Un_1[Aindex(i,j)] = 294.15
        #         else : continue
        Un=Un_1




 
        if t>tau:
            #ENERGIE CODE*****************************************************************************
            print(t)
            dT_tot = 0
            dT_up = []
            dT_down = []
            dT_side = []
            for i in range(Nz):
                for j in range(Nx):
                    if (p/d-1) > i > (p/d+1) and j <= (l_x/(2*d)+1):
                        dT_up.append((Unr[i,j] - Unr[i-1,j])/(d))
                    if ((l_z+p)/d-1)< i < ((l_z+p)/d+1) and j <= (l_x/(2*d)+1):
                        dT_down.append((Unr[i+1,j] - Unr[i,j])/(d))
                    if i >= (p/d+1)  and  i <= ((l_z+p)/d+1) and (l_x/(2*d)-1) < j < (l_x/(2*d)+1):
                        dT_side.append((Unr[i,j+1] + Unr[i,j])/(d))
            P_up = K*sum(dT_up)*(l_x*l_x)
            P_down = K*sum(dT_down)*(l_x*l_x)
            P_side = K*sum(dT_side)*(l_z*l_x)
            Energy.append(P_down + P_up + P_side)
            #ENERGIE FIN***********************************************************************************

        # Plot du graphique au temps t
        Unr = np.reshape(Un_1,(Nz,Nx),order='F')
        plot_temperature(x,z,Unr, f'temperature2d{n}')
        images.append(imageio.v2.imread(f'temperature2d{n}.png'))
        os.remove(f'temperature2d{n}.png')

        n=n+1

    print(temps_deval)
    imageio.v2.mimsave('temperatureVenus.gif', images)
    #ENERGIE CODE*****************************************************************************
    print("Energy totale = ", 2*sum(Energy)/1000, "KJ")
    #ENERGIE CODE*****************************************************************************

    print(f""" 
CALCUL TERMINÉ : ANIMATION SAUVEGARDÉE
#######################################################################################################""")
with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)

p = 1   # Profondeur de l'abris [m]
l_x = 1 # Largeur de l'abris en x [m]
l_z = 1 # Hauteur de l'abris en z [m]
Lx = 3 # Largeur du domaine [m]
Lz = 3 # Hauteur du domaine [m]
d = 0.05  # Pas de discrétisation [m]


methode_matrice_2D_temporelle(planets_constants['venus'],  p, l_x, l_z, Lx, Lz, d)