import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import yaml

################ EVALUATION DE LA MATRICE A #################

def methode_matrice_2D_A(planets_constants, p, l_x, l_z, Lx, Lz, d , sparse = True, abri=True):

    # Récupération des constantes de la planète
    C_p = planets_constants['C_p']
    K = planets_constants['K']
    rho = planets_constants['rho']
    tau = planets_constants['tau']
    Q_0 = planets_constants['Q_0']
    T_s = planets_constants['T_s']
    d_pS = planets_constants['d_pS']
    sigma = 5.67e-8 # (W/(m^2K^4)) Constante de Stefan-Boltzmann
    
    Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
    Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z

    if sparse : # Matrices pleines
     A = lil_matrix((Nx * Nz, Nx * Nz), dtype=np.double)
     M = lil_matrix((Nx * Nz, Nx * Nz), dtype=np.double)

    else : # Matrices creuses
     A=np.zeros((Nx*Nz,Nx*Nz),dtype=np.double)
     M=np.zeros((Nx*Nz,Nx*Nz),dtype=np.double)


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

def methode_matrice_2D_b(planets_constants, p, l_x, l_z, Lx, Lz, temps, d, abri=True):

    # Récupération des constantes de la planète
    C_p = planets_constants['C_p']
    K = planets_constants['K']
    rho = planets_constants['rho']
    tau = planets_constants['tau']
    Q_0 = planets_constants['Q_0']
    T_s = planets_constants['T_s']
    d_pS = planets_constants['d_pS']
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


