import numpy as np
from methode_matrice_2D import methode_matrice_2D_A
from methode_matrice_2D import methode_matrice_2D_b
from distributionInitiale import distributionInitiale
from plot_temperature import plot_temperature
import matplotlib.pyplot as plt
import imageio.v2
import os
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, diags, eye
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import yaml
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import bisect

def methode_matrice_2D_temporelle (planete, p, l_x, l_z, Lx, Lz, d ):

    with open('constants.yaml') as f:
        planets_constants = yaml.safe_load(f)

    # Définition des constantes
    C_p = planets_constants[planete]['C_p']
    K = planets_constants[planete]['K']
    rho = planets_constants[planete]['rho']
    tau = planets_constants[planete]['tau']
    Q_0 = planets_constants[planete]['Q_0']
    T_s = planets_constants[planete]['T_s']
    d_pS = planets_constants[planete]['d_pS']

    alpha=C_p*rho/K

    xi =1

    # Calcul du nombre de points en x et z
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
    z = np.linspace(0, Lz, Nz)
    x= np.linspace(0, Lx, Nx)

    abri = True # Est-ce qu'il y a un abri

    # Calcul de la distribution initiale
    U0 = distributionInitiale(planete, Nx, Nz, Lx, Lz, l_x, l_z, p, d, temps=0, abri=abri)
    
    # Définition des matrices A, M et b0

    A, M = methode_matrice_2D_A(planets_constants[planete], p, l_x, l_z, Lx, Lz, d, abri=abri)
    b0 = methode_matrice_2D_b(planets_constants[planete], p, l_x, l_z, Lx, Lz, temps=0, d=d, abri=abri)

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
        bn_1 = methode_matrice_2D_b(planets_constants[planete], p, l_x, l_z, Lx, Lz, temps=t, d=d, abri=abri)


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
        
        if t<tau:
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
            Energy.append((P_tot)*(tau))
            #ENERGIE FIN***********************************************************************************
        


        n=n+1

    imageio.v2.mimsave('temperatureEarth.gif', images)
    print("Energie totale (J/m) pour une periode de temps", (tau), "sec = ", (np.sum(np.array(Energy))/1000), "KJ/m")
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
planete = "earth"


# Energy_requise = [5147136184.3522625, 2676382727.7976074, 1856341123.0519567, 1690660812.170097, 1670248406.849412, 1693532281.4559507, 1753859347.4736502, 1863887253.6476555, 2420828896.831327]
profondeur = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5]
taille = [0.25,0.5,0.75,1,1.1,1.2,1.25,1.3,1.5,1.75,2,2.5]
# Energy_requise_t = [253642678.83114666, 618932577.5651925, 1139178171.8183033, 1739438806.6334991, 2031815701.8194437, 2363014082.3926444, 2567477299.0788608, 2702821981.553166, 3515776000.5336275, 4887323176.337782, 6621310954.071945, 15274732819.35382]

# profondeur = [0.25,0.5,0.75,1,1.1,1.2,1.25,1.3,1.5,1.75,2,2.5]

def optimisation(planete, profondeur, taille):
    tau = planets_constants[planete]['tau']
    Q_0 = planets_constants[planete]['Q_0']
    l_x = 1 # Largeur de l'abris en x [m]
    l_z = 1 # Hauteur de l'abris en z [m]
    Lx = 4 # Largeur du domaine [m]
    Lz = 4 # Hauteur du domaine [m]
    d = 0.05  # Pas de discrétisation [m]

    Energy_disponible = (Q_0*tau/(np.pi))*1000*0.19
    print("Energie disponible :", Energy_disponible, "KJ/jour" )
    
    #*************************************************************************************8
    #OPTIMISATION PROFONDEUR

    # Energy_requise = []
    # for p in profondeur:
    #     Energy_requise.append(methode_matrice_2D_temporelle(planete,  p, l_x, l_z, Lx, Lz, d))
    
    #Terre
    Energy_requise = [5147136184.3522625, 2676382727.7976074, 1856341123.0519567, 1690660812.170097, 1670248406.849412, 1693532281.4559507, 1753859347.4736502, 1863887253.6476555, 2420828896.831327]

    def fun_fit(x, a, b, c):
        return a*np.exp(-x*b+d) +c*x
    
    popt, pcov = curve_fit(fun_fit, profondeur, Energy_requise, p0=[1.66415228e+09,10,5])
    x_fit = np.linspace(0.01,3,100)
    y_fit = []
    a,b,c= popt
    for x in x_fit:
        y_fit.append(fun_fit(x,a,b,c))
    plt.plot(x_fit, y_fit, label='Fit', color='skyblue')
    plt.plot(profondeur, Energy_requise, label = "Données experimentales", color="royalblue")   
    plt.legend()
    plt.xlabel("Profondeur de l'abri (m)")
    plt.ylabel("Énergie requise (KJ/m)")
    plt.show()
    
    profondeur_min = x_fit[(y_fit.index(min(y_fit)))]
    print("Profondeur minimale = ", profondeur_min)
    #******************************************************************************8
    #OPTIMISATION TAILLE

    # p = profondeur_min
    # Energy_requise_t = []
    # for l in taille:
    #     Energy_requise_t.append(methode_matrice_2D_temporelle(planete,  p, l, l, Lx, Lz, d))
    
    #Terre
    Energy_requise_t = [253642678.83114666, 618932577.5651925, 1139178171.8183033, 1739438806.6334991, 2031815701.8194437, 2363014082.3926444, 2567477299.0788608, 2702821981.553166, 3515776000.5336275, 4887323176.337782, 6621310954.071945, 15274732819.35382]

    x_taille = np.linspace(0,2.5,100)
    y_taille = []
    for i in range(len(x_taille)):
        y_taille.append(Energy_disponible)

    def fun_fit(x, a, b, c):
        return a*np.exp(-x*b+d) +c*x
    popt, pcov = curve_fit(fun_fit, taille, Energy_requise_t)
    x_fit = np.linspace(0.01,2.5,100)
    y_fit = []
    a,b,c= popt
    for x in x_fit:
        y_fit.append(fun_fit(x,a,b,c))

    plt.plot(x_fit, y_fit, label='Fit', color='skyblue') 
    plt.plot(taille, Energy_requise_t, label = "Données experimentales", color="royalblue")
    plt.plot(x_taille,y_taille, label="Limite Energetique", color="red")
    plt.legend()
    plt.xlabel("Dimensions lx=lz (m)")
    plt.ylabel("Energie requise (KJ)")
    plt.show()
    print("Taille maximale :", bisect(lambda x: fun_fit(x,a,b,c) - Energy_disponible, -10, 10))
    
    print(Energy_requise)
    print(Energy_requise_t)
    
    
optimisation(planete,profondeur,taille)
