import numpy as np
import matplotlib.pyplot as plt

#Définition paramètre de la méthode
xi = 0.75
#Définition nb discrétisation
N = 20
#Définition du temps de simulation
time = 86400

#Input A et b _______________________________
A = np.matrix(np.zeros((N,N)))
b = np.matrix(np.zeros((N,1)))

Uo = np.matrix(np.zeros((N,1)))
alpha = 5

#Définition Matrice M
M = np.matrix(np.eye(N))
M[0,0]=0
M[N-1,N-1]=0
#Définition pas de temps
dt = 1
h = dx = dz = 0.001

#Initiation itérations 
Temptemps = []
b_time = []
# À définir en fonction du vecteur b

Un = Uo
for n in range(int(time/dt)):
  bn = b[n]
  bn_1 = b[n+1]
  A_prime = (M - (dt/(alpha*h**2))*xi*A)
  b_prime = ((M + (dt/(alpha*h**2))*(1-xi)*A)@Un - dt/(alpha*h**2)*(xi*bn_1 + (1-xi)*bn))
  Un_1 = np.linalg.solve(A_prime, b_prime)
  Un = Un_1
  Temptemps.append(Un_1)
  
