import numpy as np
import matplotlib.pyplot as plt


def methode_matrice_2D(C_p, K, rho, tau, Q_0, T_s, d_pS, p, l_x, l_z, Lx, Lz, temps, d):

     Nx=int(np.rint(Lx/d+1)) # Nombre de nœuds le long de X
     Nz=int(np.rint(Lz/d+1)) # Nombre de nœuds le long de Z
     A=np.zeros((Nx*Nz,Nx*Nz),dtype=np.double)
     b=np.zeros((Nx*Nz,1),dtype=np.double)

     
     sigma = 5.67e-8 # (W/(m^2K^4)) Constante de Stefan-Boltzmann
     def Aindex(i,j): #Associé la case i,j à sa colone dans la matrice M
          index=(j-1)*Nz+i
          return index-1
     
     def St(Q_0, d_pS, tau, temps):
        S = Q_0*(1+np.cos(temps/tau))
        return S

     for i in np.arange(1,Nz+1,1): #i=1,..,Ny - numérotation des nœuds sur un maillage physique
        z=np.round((i-1)*d, decimals=12)
        for j in np.arange(1,Nx+1,1): #j=1,..,Nx - numérotation des nœuds sur un maillage physique
            x=np.round((j-1)*d, decimals=12)


            Aij = Aindex(i,j)

            # Condition frontière en z==0
            if i == 1:
              A[Aindex(i,j),Aindex(i,j)] = -(3+2*d/K*T_s**3)
              A[Aindex(i,j),Aindex(i+1,j)] = 4
              A[Aindex(i,j),Aindex(i+2,j)] = -1
              b[Aindex(i,j)] = -(2*d)*St(Q_0, d_pS, tau, temps)/(K)

            # Condition frontière en z==Lz
            elif i == Nz:
              A[Aindex(i,j), Aindex(i,j)] = 1    
              b[Aindex(i,j)] = T_s
            # Condition frontière en x==0 (T'(0) = 0)
            elif j == 1 :
              A[Aindex(i,j),Aindex(i,j)] = -3
              A[Aindex(i,j),Aindex(i,j+1)] = 4
              A[Aindex(i,j),Aindex(i,j+2)] = -1

            # Condition frontière en x==Lx (T'(Lx) = 0)
            elif j == Nx:
              A[Aindex(i,j),Aindex(i,j)] = 3
              A[Aindex(i,j),Aindex(i,j-1)] = -4
              A[Aindex(i,j),Aindex(i,j-2)] = 1              

            # Temperature à 394 K pour tous les points dans l'abris
            elif i > p/d+1  and i < (l_z+p)/d+1 and j < l_x/(2*d)+1:
              A[Aindex(i,j), Aindex(i,j)] = 1
              b[Aindex(i,j)] = 394 

            # Tous les autres points
            elif j > 1 and j < Nx and i > 1 and i < Nz  : 
              A[Aindex(i,j),Aindex(i,j-1)] = 1
              A[Aindex(i,j),Aindex(i,j+1)] = 1
              A[Aindex(i,j),Aindex(i,j)] = -4
              A[Aindex(i,j),Aindex(i-1,j)] = 1
              A[Aindex(i,j),Aindex(i+1,j)] = 1
            else :
               print('indice ne rentre dans aucune catégorie')
     
     
     return A, b, x, z, Nx, Nz

A, b, x, z, Nx, Ny = methode_matrice_2D(C_p=675, K=1, rho=2000, tau=12, Q_0=492, T_s=288.15, d_pS=1, p=3, l_x=3, l_z=3, Lx=4, Lz=4, temps = 0, d =1)

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

