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
        dt = alpha*d**2

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
            Energy.append(2*(P_tot)*(tau)) #facteur 2 car demi-abri
            #ENERGIE FIN***********************************************************************************
        


        n=n+1

    imageio.v2.mimsave('temperatureEarth.gif', images)
    print("Energie totale (J/m) pour une periode de temps", (tau), "sec = ", (np.sum(np.array(Energy))/1000), "KJ/m")
    print(f""" 
CALCUL TERMINÉ : ANIMATION SAUVEGARDÉE
#######################################################################################################""")
    return np.sum(np.array(Energy))/1000


with open('constants.yaml') as f:
    planets_constants = yaml.safe_load(f)

p = 1   # Profondeur de l'abris [m]
l_x = 15 # Largeur de l'abris en x [m]
l_z = 10 # Hauteur de l'abris en z [m]
Lx = 20 # Largeur du domaine [m]
Lz = 20 # Hauteur du domaine [m]
d = 0.05  # Pas de discrétisation [m]
planete = "earth"
tau = planets_constants[planete]['tau']
Q_0 = planets_constants[planete]['Q_0']



Energy_disponible = (Q_0*tau/(np.pi))*1000*0.19
print("Energie disponible :", Energy_disponible, "KJ/jour")

# print(methode_matrice_2D_temporelle(planete, p, l_x, l_z, Lx, Lz, d))

d_list = [0.15,0.17,0.2,0.25,0.3,0.5,0.75]
# Energie_list = [65610382.91315033, 43173995.03670958, 31558401.838820755, 39451633.83497372, 208487013.7098127, 176145684.42017445, 153719620.44568586, 139583791.73848617]

Energie_list = [65610382.91315033, 59670447.64978681, 43173995.03670958, 31558401.838820755, 39451633.83497372, 139583791.73848617, 322522762.3537608] 

#lx = 15, lz =10, Lx = Lz = 20
# Energie_list = [196475307.62077984, 131671544.4179855, 95308964.76084466, 31558401.838820755, 139583791.73848617, 204576910.76623088]
#autres dim
#[16731666.046718024, 4023573.1418389753, 2821721.607497795, 1901417.384075963, 629413.6175423791, 1109636.5816942938]
#d_list = [0.08,0.1,0.2,0.3,0.5,0.75,1,2]
#[124380698.53162856, 95308964.76084466, 43173995.03670958, 39451633.83497372, 139583791.73848617, 322522762.3537608, 204576910.76623088, 51394814.59535041] 

# for d in d_list:
#     print("pas", d)
#     Energie_list.append(methode_matrice_2D_temporelle(planete, p, l_x, l_z, Lx, Lz, d))
Error = []
for i in range(len(Energie_list)-1):
    Error.append((abs(Energie_list[i+1]-Energie_list[i])/Energie_list[i])*100)

d_list = [0.15,0.17,0.2,0.25,0.3,0.5]

# Error_log = [np.log(0.09053346436380849), np.log(0.27645934064207023), np.log(0.2690414261643479), np.log(0.250115073521984), np.log(2.53809913988266), np.log(1.3106032465289057)]
def fun_fit(x, A,b):
    return A*x**b

popt, pcov = curve_fit(fun_fit, d_list, Error)
x_fit = np.linspace(0,1,100)
y_fit = []
A,p= popt
for x in x_fit:
    y_fit.append(fun_fit(x,A,p))


# d_list = d_list = [0.15,0.2,0.25,0.3,0.35,0.4,0.45]


plt.loglog(x_fit,y_fit,'--',label="Fit")
plt.loglog(d_list, Error, 'o', label="Données d'erreur", color = 'darkblue')
# plt.loglog(d_list,Energie_list)
plt.legend()
plt.xlabel("pas de discrétisation")
plt.ylabel("Erreur%")
plt.show()





profondeur = [0.25,0.6,1,1.3,1.7,2]
largeur  = [1,2,3,4,5,10]



def optimisation(planete, profondeur, largeur):
    tau = planets_constants[planete]['tau']
    Q_0 = planets_constants[planete]['Q_0']
    l_x = 1 # Largeur de l'abris en x [m]
    l_z = 5 # Hauteur de l'abris en z [m]
    Lx = 20 # Largeur du domaine [m]
    Lz = 20  # Hauteur du domaine [m]
    d = 0.1  # Pas de discrétisation [m]

    Energy_disponible = (Q_0*tau/(np.pi))*1000*0.19
    print("Energie disponible :", Energy_disponible, "KJ/jour")
    
    #*************************************************************************************8
    #OPTIMISATION PROFONDEUR
    # Energy_requise = []
    # for p in profondeur:
    #     Energy_requise.append(methode_matrice_2D_temporelle(planete,  p, l_x, l_z, Lx, Lz, d))
    ##Terre
    # Energy_requise = [942627918.0116413, 260224745.95214584, 95308964.76084466, 84187696.88203554, 75922505.7547069, 70974485.12136276]
    ##Mars
    # Energy_requise = [280015361.70601636, 72252348.41086335, 78243119.09877847, 71135755.55708829, 64576797.532311894, 60551616.98685229]
    #Venus 
    # Energy_requise = [119186284629.17044, 116218753450.68849, 107824499576.25157, 102732135359.23335, 97581659024.5799, 93946272080.23279, 93475177087.7238, 93776791450.56203, 93963149703.15579, 96514118249.72557, 99984616834.86806, 109336983164.63033, 122873801140.39725]
    # [1.66415228e+09,10,555, 8]
    #Europe
    Energy_requise = [471436019.0085587, 472688656.3120705, 472455658.91815466, 480262880.28744435, 483877583.93724173, 488643013.71045154]
    
    print(Energy_requise)
    
    def fun_fit(x, a, b, c, d):
        return a*np.exp(-x*b+d) +c*x
    
    
    popt, pcov = curve_fit(fun_fit, profondeur, Energy_requise,p0=[1.66415228e+09,10,555, 8])
    x_fit = np.linspace(0.01,6,100)
    y_fit = []
    a,b,c,d= popt
    for x in x_fit:
        y_fit.append(fun_fit(x,a,b,c,d))
    plt.plot(x_fit, y_fit, label='Fit', color='thistle')
    plt.plot(profondeur, Energy_requise, label = "Données experimentales", color="mediumorchid")   
    plt.legend()
    plt.xlabel("Profondeur de l'abri (m)")
    plt.ylabel("Énergie requise (KJ/m)")
    plt.show()
    
    profondeur_min = x_fit[(y_fit.index(min(y_fit)))]
    print("Profondeur minimale = ", profondeur_min)
    
    #******************************************************************************8
    #OPTIMISATION PROFONDEUR
    tau = planets_constants[planete]['tau']
    Q_0 = planets_constants[planete]['Q_0']
    l_x = 1 # Largeur de l'abris en x [m]
    l_z = 5 # Hauteur de l'abris en z [m]
    Lx = 50 # Largeur du domaine [m]
    Lz = 50  # Hauteur du domaine [m]
    d = 0.5  # Pas de discrétisation [m]
    p = profondeur_min
    
    # Energy_requise_t = []
    # for lx in largeur:
    #     Energy_requise_t.append(methode_matrice_2D_temporelle(planete,  p, lx,l_z, Lx, Lz, d))
    ##Terre
    # Energy_requise_t = [1985970895.4919682, 6501747445.492055, 23950335090.794014, 52606796384.12155, 92464815246.6938]
    #Mars
    # Energy_requise_t = [8460416.018684648, 24312306.17983813, 85302659.05467126, 185213177.63627484, 323988901.744541]
    #Venus 
    # Energy_requise_t =[5849914393.710241, 8737551562.17036, 18463857441.52574, 33102588132.738346, 52486775311.47175]
    #Europe
    # Energy_requise_t = [1339696506.21129, 4382312936.379228, 16147678139.02944, 35472282112.57271, 62349748760.45685]
    Energy_requise_t = [1210366.3437502955, 306048713.56937945, 531625279.5261729, 837550700.8609563, 1219670911.3887753, 4268229911.92946]
    print(Energy_requise_t)
    for i in range(len(Energy_requise_t)):
        Energy_requise_t[i] = Energy_requise_t[i]*2
    
    x_taille = np.linspace(0,10,100)
    y_taille = []
    for i in range(len(x_taille)):
        y_taille.append(Energy_disponible)

    def fun_fit(x, a, b, c,d):
        return a*x**2+b*x+c
    popt, pcov = curve_fit(fun_fit, largeur, Energy_requise_t, p0=[5,5,1,-100])
    x_fit = np.linspace(0,10,100)
    y_fit = []
    a,b,c,d= popt
    for x in x_fit:
        y_fit.append(fun_fit(x,a,b,c,d))
    
    plt.title("profondeur = 0.192 m, lz = 5 m, ly = 2m")
    plt.plot(x_fit, y_fit, label='Fit', color='thistle') 
    plt.plot(largeur, Energy_requise_t, label = "Données experimentales", color="mediumorchid")
    plt.plot(x_taille,y_taille, label="Limite Energetique", color="red")
    plt.legend()
    plt.xlabel("Dimensions lx (m)")
    plt.ylabel("Energie requise (KJ)")
    plt.show()
    
    print(Energy_requise)
    print(Energy_requise_t)
    
    print("Profondeur minimale : ", profondeur_min)
    print("Taille maximale :", bisect(lambda x: fun_fit(x,a,b,c,d) - Energy_disponible, 0, 100))

# optimisation(planete,profondeur,largeur)

#taille ini = 1,45
# Profondeur minimale :  1.0670707070707073
# Taille maximale : 1.253667617783094
# [10104463262.594048, 5143469865.330789, 3514523616.659591, 3227858420.91325, 3258620461.0897565, 3416939243.6499968, 3730256265.0281262, 4340810091.311294] 
# [255800110.1703819, 624155222.5581789, 1147411224.320585, 1747478636.417749, 2568709130.3875213, 3495834204.5919585, 4811601858.10958, 6415737398.175278] 

#Taille ini = 1.2
# Profondeur minimale :  1.1274747474747475
# Taille maximale : 1.2545193661674148
# [7086275257.543439, 3633888479.853899, 2507425254.7376704, 2291014389.2737207, 2280917007.17665, 2339982686.294331, 2466535704.4711103, 2696369176.805922]  
# [254604642.48445806, 621188476.9522421, 1142516295.6240325, 1742120561.7774243, 2565834672.336313, 3502236997.5518165, 4843293783.528274, 6506893695.683995]


#Taille ini =1
# Profondeur minimale :  1.158
# Taille maximale : 1.252

#Terre
    # Energy_requise = [5147136184.3522625, 2676382727.7976074, 1856341123.0519567, 1690660812.170097, 1670248406.849412, 1693532281.4559507, 1753859347.4736502, 1863887253.6476555, 2420828896.831327]
    #Mars
    # Energy_requise = [1610008083.4070761, 1336093278.9603126, 1443221584.8115158, 1433954521.0068386, 1431381751.937218, 1455322251.4651198, 1434055580.490696, 1472477195.8403072, 1458798444.7375956, 1513924689.4273956, 1611555399.7170882, 2098253513.9943404]
    
    
    #Terre
    # Energy_requise_t = [253642678.83114666, 618932577.5651925, 1139178171.8183033, 1739438806.6334991, 2031815701.8194437, 2363014082.3926444, 2567477299.0788608, 2702821981.553166, 3515776000.5336275, 4887323176.337782, 6621310954.071945, 15274732819.35382]
    #Mars
    # Energy_requise_t = [221978432.02442053, 539885892.7143496, 988130286.698987, 1496522350.2646198, 1740856329.1225133, 2015139144.6863348, 2183192808.0959454, 2292335168.0213666, 2941685637.0081844, 3988966079.5682507, 5201887040.404957, 9576369796.97882]
    
