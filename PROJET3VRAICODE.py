import numpy as np
import matplotlib.pyplot as plt
import imageio.v2
import os
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, diags, eye
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import yaml
from tqdm import tqdm
from memory_profiler import profile
import psutil
from scipy.optimize import curve_fit

# Define the function to fit
def fit(x, a, b,):
    return a * x**b

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss


def distributionInitiale (Nx, Nz, Lx, Lz, Ts):
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

def plot_temperature (x, z, Tr, label):
    plt.pcolor(x, z, Tr, cmap='hot', vmin=280, vmax=320)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title('T$_0$(x,y) [K]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().invert_yaxis()
    plt.savefig(f'{label}.png')

################ EVALUATION DE LA MATRICE A #################

def methode_matrice_2D_A(p, l_x, l_z, Lx, Lz, d , sparse = True, abri=True):

    # Récupération des constantes de la planète
    C_p= 675      # Capacité thermique [J/kgK]
    K= 1          # Conductivité thermique [W/mK]
    rho= 2000     # Masse volumique [kg/m^3]
    tau= 83600    # Période de rotation [s]
    Q_0= 492      # Rayonnement solaire [W/m^2]
    T_s= 305.2    # Température moyenne [K]
    d_pS= 1       # Distance Planète-Soleil [AU]
    sigma = 5.67e-8 # (W/(m^2K^4)) Constante de Stefan-Boltzmann
    
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z

    if sparse : # Matrices pleines
     A = lil_matrix((Nx * Nz, Nx * Nz), dtype=np.double)
     M = lil_matrix((Nx * Nz, Nx * Nz), dtype=np.double)

    else : # Matrices creuses
     A=np.zeros((Nx*Nz,Nx*Nz),dtype=np.double)
        
    def Aindex(i,j): #Associé la case i,j à sa colone dans la matrice M
         index=(j-1)*Nz+i
         return index-1
    
    for i in np.arange(1,Nz+1,1): #i=1,..,Nz - numérotation des nœuds sur un maillage physique
       z=np.round((i-1)*d, decimals=12)
       for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
           x=np.round((j-1)*d, decimals=12)

          #  # Condition frontière en x==Lx (T'(Lx) = 0)
           if j == Nx:
             A[Aindex(i,j),Aindex(i,j)] = 3
             A[Aindex(i,j),Aindex(i,j-1)] = -4
             A[Aindex(i,j),Aindex(i,j-2)] = 1
             M[Aindex(i,j),Aindex(i,j)] = 0

           # Condition frontière en z==0s
           elif i == 1:
             A[Aindex(i,j),Aindex(i,j)] = -(3+(2*d*sigma/K)*T_s**3)
             A[Aindex(i,j),Aindex(i+1,j)] = 4
             A[Aindex(i,j),Aindex(i+2,j)] = -1
             M[Aindex(i,j),Aindex(i,j)] = 0

           # Condition frontière en z==Lz
           elif i == Nz:
             A[Aindex(i,j), Aindex(i,j)] = 1
             M[Aindex(i,j),Aindex(i,j)] = 0
           
           # Condition frontière en x==0 (T'(0) = 0)
           elif j == 1 :
             A[Aindex(i,j),Aindex(i,j)] = -3
             A[Aindex(i,j),Aindex(i,j+1)] = 4
             A[Aindex(i,j),Aindex(i,j+2)] = -1
             M[Aindex(i,j),Aindex(i,j)] = 0
           
          #  Temperature à 394 K polur tous les points dans l'abris
           elif i > p/d+1  and i < (l_z+p)/d+1 and j < l_x/(2*d)+1 and abri:
             A[Aindex(i,j), Aindex(i,j)] = 1
             M[Aindex(i,j),Aindex(i,j)] = 0

           
           # Tous les autres points
           elif j > 1 and j < Nx and i > 1 and i < Nz  :
             A[Aindex(i,j),Aindex(i,j-1)] = 1
             A[Aindex(i,j),Aindex(i,j+1)] = 1
             A[Aindex(i,j),Aindex(i,j)] = -4
             A[Aindex(i,j),Aindex(i-1,j)] = 1
             A[Aindex(i,j),Aindex(i+1,j)] = 1
             M[Aindex(i,j),Aindex(i,j)] = 1
           
           else :
              print('indice ne rentre dans aucune catégorie')

    return A, M

################### EVALUATION DU VECTEUR b ############################

def methode_matrice_2D_b(p, l_x, l_z, Lx, Lz, temps, d, abri=True):

    # Récupération des constantes de la planète
    C_p= 675      # Capacité thermique [J/kgK]
    K= 1          # Conductivité thermique [W/mK]
    rho= 2000     # Masse volumique [kg/m^3]
    tau= 83600    # Période de rotation [s]
    Q_0= 492      # Rayonnement solaire [W/m^2]
    T_s= 305.2    # Température moyenne [K]
    d_pS= 1       # Distance Planète-Soleil [AU]
    sigma = 5.67e-8 # (W/(m^2K^4)) Constante de Stefan-Boltzmann

    
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z

    b=np.zeros((Nx*Nz,1),dtype=np.double)
    
    # Définition de la source de chaleur S(t)=Q_0(1+cos(2pi*t/tau))
    def St(Q_0, d_pS, tau, temps):
       S = Q_0 *(1+np.sin(temps*2*np.pi/tau))
       return S
    
    def Aindex(i,j): #Associé la case i,j à sa colone dans la matrice M
         index=(j-1)*Nz+i
         return index-1
    
    for i in np.arange(1,Nz+1,1): #i=1,..,Nz - numérotation des nœuds sur un maillage physique
       z=np.round((i-1)*d, decimals=12)
       for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
           x=np.round((j-1)*d, decimals=12)

          #  # Condition frontière en x==Lx (T'(Lx) = 0)
           if j == Nx:
            continue
           
           # Condition frontière en z==0s
           elif i == 1:
             b[Aindex(i,j)] = -(2*d)*St(Q_0, d_pS, tau, temps)/(K)

           # Condition frontière en z==Lz
           elif i == Nz:
             b[Aindex(i,j)] = T_s
           
           # Condition frontière en x==0 (T'(0) = 0)
           elif j == 1 :
            continue
           
          #  Temperature à 394 K polur tous les points dans l'abris
           elif i > p/d+1  and i < (l_z+p)/d+1 and j < l_x/(2*d)+1 and abri:
             b[Aindex(i,j)] = 294.15
           
           # Tous les autres points
           elif j > 1 and j < Nx and i > 1 and i < Nz  :
             continue
           
           else :
              print('indice ne rentre dans aucune catégorie')

    return b

def methode_matrice_2D_temporelle (p, l_x, l_z, Lx, Lz, d ):
    time_ini =time.time_ns()
    # Définition des constantes
    
    C_p= 675      # Capacité thermique [J/kgK]
    K= 1          # Conductivité thermique [W/mK]
    rho= 2000     # Masse volumique [kg/m^3]
    tau= 83600    # Période de rotation [s]
    Q_0= 492      # Rayonnement solaire [W/m^2]
    T_s= 305.2    # Température moyenne [K]
    d_pS= 1       # Distance Planète-Soleil [AU]
    sigma = 5.67e-8 # (W/(m^2K^4)) Constante de Stefan-Boltzmann
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
    # print(U0)
    # def Uindex(i,j): #Associé la case i,j à sa colone dans la matrice M
    #     index=(j-1)*Nz+i
    #     return index-1
    # for i in range(Nz+1): #i=1,..,Nz - numérotation des nœuds sur un maillage physique
    #    for j in range(Nx+1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
    #        if i > p/d+1  and i < (l_z+p)/d+1 and j < l_x/(2*d)+1:
    #             U0r[Uindex(i,j), Uindex(i,j)] = 294
           

    plot_temperature(x,z,U0r, 'distribInitiale') # Graphique de Distribution initiale

    # Définition des matrices A, M et b0
    abri = True # Est-ce qu'il y a un abri

    A, M = methode_matrice_2D_A(p, l_x, l_z, Lx, Lz, d, abri=abri)
    b0 = methode_matrice_2D_b(p, l_x, l_z, Lx, Lz, temps=0, d=d, abri=abri)
    # Définition du pas de temps
    dt = np.min([alpha*d**2/4, tau/50])
    nb_iterations = 100
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
        bn_1 = methode_matrice_2D_b(p, l_x, l_z, Lx, Lz, temps=t, d=d, abri=abri)


        b_prime_1 = (M + dt*(1-xi)/(alpha*d**2)*A).dot(Un).flatten()
        b_prime_2 = (dt/(alpha*d**2)*(xi*bn_1+(1-xi)*bn)).flatten()

        b_prime = b_prime_1 - b_prime_2

        Un_1 = spsolve(A_prime, b_prime)
        Un=Un_1
        
        
        # Plot du graphique au temps t
        Unr = np.reshape(Un_1,(Nz,Nx),order='F')
        
        Energy = []
        if t>tau:
            #ENERGIE CODE*****************************************************************************
            dT_tot = 0
            for i in range(Nz):
                for j in range(Nx):
                    if (p/d-1) > i > (p/d+1) and j <= (l_x/(2*d)+1):
                        dT_up = (Unr[i,j] - Unr[i-1,j])/(d)
                        dT_tot += abs(dT_up)
                    if ((l_z+p)/d-1)< i < ((l_z+p)/d+1) and j <= (l_x/(2*d)+1):
                        dT_down = (Unr[i+1,j] - Unr[i,j])/(d)
                        dT_tot += abs(dT_down)
                    if i >= (p/d+1)  and  i <= ((l_z+p)/d+1) and (l_x/(2*d)-1) < j < (l_x/(2*d)+1):
                        dT_side = (Unr[i,j+1] + Unr[i,j])/(d)
                        dT_tot += abs(dT_side)
        
            Energy.append((2*l_x+l_z)*dT_tot)
            #ENERGIE FIN***********************************************************************************
        
        plt.clf()
        plot_temperature(x,z,Unr, f'temperature2d{n}')
        images.append(imageio.v2.imread(f'temperature2d{n}.png'))
        os.remove(f'temperature2d{n}.png')
        
        n=n+1

    imageio.v2.mimsave('temperature2d.gif', images)
    time_final = time.time_ns()
    #ENERGIE CODE*****************************************************************************
    print("Energy totale = ", 2*K*sum(Energy))
    #ENERGIE CODE*****************************************************************************
    print(f""" 
CALCUL TERMINÉ : ANIMATION SAUVEGARDÉE
#######################################################################################################""")


p = 3   # Profondeur de l'abris [m]
l_x = l_z = 3 # Largeur de l'abris en x [m] # Hauteur de l'abris en z [m]
Lx = Lz = 10 # Largeur du domaine [m] et  Hauteur du domaine [m]
d = 0.1  # Pas de discrétisation [m]


d_list = [1, 0.5, 0.3, 0.1, 0.05]
t_list = []
mem_list = []
if __name__ == '__main__':
    for i in d_list:
        before_memory = 0
        before_time = time.time_ns()
        methode_matrice_2D_temporelle(p, l_x, l_z, Lx, Lz, i)
        after_memory = get_memory_usage()
        after_time = time.time_ns()
        peak_time = (after_time-before_time)/1.0e9
        peak_memory = after_memory - before_memory
        t_list.append(peak_time)
        mem_list.append(peak_memory)
        print(f"Peak memory usage: {peak_memory/1024} KB")
        print(f"Time: {peak_time} seconds")
        


plt.plot(d_list, t_list)
plt.legend()
plt.xlabel('pas de discrétisation')
plt.ylabel('temps de calcul (s)')
plt.title('Temps de calcul en fonction du pas de discrétisation')
plt.show()

poptmem, pcov = curve_fit(fit, (d_list), (mem_list))
a_mem, b_mem = poptmem
mem_list_fit = fit(d_list, a_mem, b_mem)
    
plt.loglog((d_list), (mem_list))
plt.loglog((d_list), mem_list_fit, label = "fit lineaire: $t_{{ini}} = {a_mem:.2f})\delta^{{{b_mem:.2f}}}$")
plt.legend()
plt.xlabel('pas de discrétisation')
plt.ylabel('mémoire utilisée (kB)')
plt.title('Mémoire utilisée en fonction du pas de discrétisation')
plt.show()

popt, pcov = curve_fit(fit, (d_list), (t_list))
a_t, b_t = popt
t_list_fit = fit(d_list, a_t, b_t)
    
plt.loglog((d_list),(t_list))
plt.loglog((d_list), t_list_fit, label = "fit lineaire: $t_{{ini}} = ({a_t:.2e})\delta^{{{b_t:.2f}}}$")
plt.legend()
plt.xlabel('pas de discrétisation')
plt.ylabel('temps de calcul (s)')
plt.title('Temps de calcul en fonction du pas de discrétisation')
plt.show()
