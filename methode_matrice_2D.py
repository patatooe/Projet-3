import numpy as np



def methode_matrice_2D(C_p, K, rho, tau, Q_0, T_s, d_pS, p, l_x, l_z, Lx, Lz, temps, d):


     def Mindex(i,j): #Associé la case i,j à sa colone dans la matrice M
          index=(j-1)+i
          return index-1

     Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
     Ny=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
     A=np.zeros((Nx*Ny,Nx*Ny),dtype=np.double)
     b=np.zeros((Nx*Ny,1),dtype=np.double)
     T=np.zeros((Nx*Ny,1),dtype=np.double)
     Tr=np.zeros((Ny,Nx),dtype=np.double)


     for i in np.arange(1,Ny+1,1): #i=1,..,Ny - numérotation des nœuds sur un maillage physique
        z=np.round((i-1)*d, decimals=12)
        for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
            x=np.round((j-1)*d, decimals=12)

     


     
     
     return 1
