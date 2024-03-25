import numpy as np
import matplotlib.pyplot as plt


def methode_matrice_2D(C_p, K, rho, tau, Q_0, T_s, d_pS, p, l_x, l_z, Lx, Lz, temps, d):

     sigma = 5.67e-8 # (W/(m^2K^4)) Constante de Stefan-Boltzmann
     def Aindex(i,j): #Associé la case i,j à sa colone dans la matrice M
          index=(j-1)+i
          return index-1
     
     def St(Q_0, d_pS, tau, temps):
        S = Q_0*(1+np.cos(temps/tau))

     Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
     Ny=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
     A=np.zeros((Nx*Ny,Nx*Ny),dtype=np.double)
     b=np.zeros((Nx*Ny,1),dtype=np.double)


     for i in np.arange(1,Ny+1,1): #i=1,..,Ny - numérotation des nœuds sur un maillage physique
        z=np.round((i-1)*d, decimals=12)
        for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
            x=np.round((j-1)*d, decimals=12)

        # Condition frontière en x==0 (T'(0) = 0)
        if x == 0 :
          A[Aindex(i,j),Aindex(i,1)] = -3
          A[Aindex(i,j),Aindex(i,2)] = 4
          A[Aindex(i,j),Aindex(i,3)] = -1
        
        # Condition frontière en x==Lx (T'(Lx) = 0)
        elif x == Lx:
          A[Aindex(i,j),Aindex(i,1)] = -3
          A[Aindex(i,j),Aindex(i,2)] = 4
          A[Aindex(i,j),Aindex(i,3)] = -1              

        # Condition frontière en z==0
        elif z == 0:
          A[Aindex(i,j),Aindex(i,1)] = -(K/(2*d)*sigma*T_s^3)
          A[Aindex(i,j),Aindex(i,2)] = 4
          A[Aindex(i,j),Aindex(i,3)] = -1
          b[Aindex(i,j)] = -K*St(Q_0, d_pS, tau, temps)/(2*d)

        # Condition frontière en z==Lz
        elif z == Lz:
          A[Aindex(i,j), Aindex(i,j)] = 1    
          b[Aindex(i,j)] = T_s
        
        # Temperature à 394 K pour tous les points dans l'abris
        elif z > p  and z < l_z+p and x < l_x/2:
          A[Aindex(i,j), Aindex(i,j)] = 1
          b[Aindex(i,j), Aindex(i,j)] = 394 

        # Tous les autres points
        elif x > 0 and x < Lx and z > 0 and z < Lz  : 
          A[Aindex(i,j),Aindex(i,j-1)] = 1
          A[Aindex(i,j),Aindex(i,j+1)] = 1
          A[Aindex(i,j),Aindex(i,j)] = -4
          A[Aindex(i,j),Aindex(i-1,j)] = 1
          A[Aindex(i,j),Aindex(i+1,j)] = 1
     
     
     return A, b, x, z, Nx, Ny

A, b, x, z, Nx, Ny = methode_matrice_2D(C_p=675, K=1, rho=2000, tau=12, Q_0=492, T_s=288.15, d_pS=1, p=3, l_x=3, l_z=3, Lx=10, Lz=10, temps = 0, d =10)

T=np.zeros((Nx*Ny,1),dtype=np.double)
Tr=np.zeros((Ny,Nx),dtype=np.double)

T = np.linalg.solve(A, b)
Tr=np.reshape(T,(Ny,Nx),order='F')

plt.figure(3)
plt.pcolor(x,z,Tr)
plt.colorbar(mappable=None, cax=None, ax=None)
plt.title('T(x,y) [K]')
plt.xlabel('x [m]')    
plt.ylabel('z [m]')
plt.show()

