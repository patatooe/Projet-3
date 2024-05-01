import numpy as np
from plot_temperature import plot_temperature
import matplotlib.pyplot as plt
import time
import yaml
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import bisect
from methode_matrice_2D_temporelle import methode_matrice_2D_temporelle



# Energy_requise = [5147136184.3522625, 2676382727.7976074, 1856341123.0519567, 1690660812.170097, 1670248406.849412, 1693532281.4559507, 1753859347.4736502, 1863887253.6476555, 2420828896.831327]
# Energy_requise_t = [253642678.83114666, 618932577.5651925, 1139178171.8183033, 1739438806.6334991, 2031815701.8194437, 2363014082.3926444, 2567477299.0788608, 2702821981.553166, 3515776000.5336275, 4887323176.337782, 6621310954.071945, 15274732819.35382]

# profondeur = [0.25,0.5,0.75,1,1.1,1.2,1.25,1.3,1.5,1.75,2,2.5]

def optimisation(planete, profondeur, taille, p, l_x, l_z, Lx, Lz,d):

    
    with open('constants.yaml') as f:
        planets_constants = yaml.safe_load(f)

    tau = planets_constants[planete]['tau']
    Q_0 = planets_constants[planete]['Q_0']

    Energy_disponible = (Q_0*tau/(np.pi))*1000*0.19
    print("Energie disponible :", Energy_disponible, "KJ/jour" )
    
    #*************************************************************************************8
    #OPTIMISATION PROFONDEUR

    Energy_requise = []
    for p in profondeur:
         Energy_requise.append(methode_matrice_2D_temporelle(planete,  p, l_x, l_z, Lx, Lz, d))
    
    #Terre
    #Energy_requise = [5147136184.3522625, 2676382727.7976074, 1856341123.0519567, 1690660812.170097, 1670248406.849412, 1693532281.4559507, 1753859347.4736502, 1863887253.6476555, 2420828896.831327]

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

    p = profondeur_min
    Energy_requise_t = []
    for l in taille:
         Energy_requise_t.append(methode_matrice_2D_temporelle(planete,  p, l, l, Lx, Lz, d))
    
    #Terre
    #Energy_requise_t = [253642678.83114666, 618932577.5651925, 1139178171.8183033, 1739438806.6334991, 2031815701.8194437, 2363014082.3926444, 2567477299.0788608, 2702821981.553166, 3515776000.5336275, 4887323176.337782, 6621310954.071945, 15274732819.35382]

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
    

#---------------------------------------------------------test optimisation-----------------------

l_x = 15 # Largeur de l'abris en x [m]
l_z = 10 # Hauteur de l'abris en z [m]
Lx = 50 # Largeur du domaine [m]
Lz = 50 # Hauteur du domaine [m]
d = 0.05  # Pas de discrétisation [m] 
planete = 'earth'
profondeur = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5]
taille = [0.25,0.5,0.75,1,1.1,1.2,1.25,1.3,1.5,1.75,2,2.5]

optimisation(planete,profondeur,taille, l_x, l_z, Lx, Lz,d)
