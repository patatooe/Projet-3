import numpy as np
import matplotlib.pyplot as plt
from methode_matrice_2D import methode_matrice_2D
import imageio.v2
import os
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time


#Définition paramètre de la méthode
xi = 0.75
#Définition du temps de simulation
time = 86400

# Définition des constantes des planètes
C_p = 675  # [J/(kgK)]
k=1        # [W/(mK)] 
rho=2000   # [Kg/m^3]
tau=5000   # [s]
Q_0=492    # [W/m^2]
T_s=305.2    # [K]
d_pS=1

# Variables pouvant changer
p=3 # (m)
l_x=3 # (m)
l_z=3 #(m)
Lx=1 #(m)
Lz=10 #(m)
temps=0 #(s)
d=0.1 #(m)


#Input A et b _______________________________
# A = np.matrix(np.zeros((N,N)))
# b = np.matrix(np.zeros((N,1)))

A0, b0, Nx, Nz = methode_matrice_2D(C_p=C_p, K=k, rho=rho, tau=tau, Q_0=Q_0, T_s=T_s, d_pS=d_pS, p=p, l_x=l_x, l_z=l_z, Lx=Lx, Lz=Lz, temps = temps, d =d)


Uo = np.matrix(np.full((Nx*Nz,1), 2*T_s))
alpha = C_p*rho/k

#Définition Matrice M
M = np.matrix(np.eye(Nx*Nz))
M[0,0]=0
M[Nx*Nz-1,Nx*Nz-1]=0
#Définition pas de temps
dt = 1
h = dx = dz = d

#Initiation itérations 
Temptemps = []
b_time = []
# À définir en fonction du vecteur b


dt = 10000
nb_iterations = 20
temps_eval = dt*nb_iterations
print(f'Critère de convergence : dt<{alpha*h**2:.2f} \nPas de temps dt = {dt} \nCritère respecté : {dt<alpha*h**2:.2f} ')
Un = Uo
bn = b0
images = []

tspace = np.linspace(0, temps_eval, dt)

n=0
for t in tspace:
  A, bn_1, Nx, Nz = methode_matrice_2D(C_p=C_p, K=k, rho=rho, tau=tau, Q_0=Q_0, T_s=T_s, d_pS=d_pS, p=p, l_x=l_x, l_z=l_z, Lx=Lx, Lz=Lz, temps = t, d =d, sparse=True)
  A_prime = (M - (dt/(alpha*h**2))*xi*A)
  b_prime = ((M + (dt/(alpha*h**2))*(1-xi)*A)@Un - dt/(alpha*h**2)*(xi*bn_1 + (1-xi)*bn))
  Un_1 = np.linalg.solve(A_prime, b_prime)

  Un = Un_1
  bn=bn_1
  Temptemps.append(Un_1)

  z = np.linspace(0, Lz, Nz)
  x= np.linspace(0, Lx, Nx)
  Tr=np.reshape(Un,(Nz,Nx),order='F') # Convertion du vecteur colone de température en matrice dépendant de la position : T_ij->T(x,y)
  plt.clf()
  plt.pcolor(x,z,np.array(Tr), vmin=300, vmax=625)
  plt.colorbar(mappable=None, cax=None, ax=None)
  plt.title('T(x,y) [K]')
  plt.xlabel('x [m]')
  plt.ylabel('z [m]')
  plt.gca().invert_yaxis()
  plt.savefig(f'temperature2d{n}.png')
  images.append(imageio.v2.imread(f'temperature2d{n}.png'))
  os.remove(f'temperature2d{n}.png')
  n+=1

imageio.v2.mimsave('temperature2d.gif', images)
  
