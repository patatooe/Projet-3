import numpy as np
import matplotlib.pyplot as plt

def calcul_energie_chauffage(lbd, emissivite, Qo, A, T_surf, T_grad, dt,t):
     
     #flux rayonnemment 
     alpha = 5.6703*10**(-8)
     F_ray = emissivite * alpha * T_surf
     
     #flux conduction 
     F_cond = lbd * T_grad
     
     #flux du Soleil
     Flux_soleil = Qo + A*np.sin(2*np.pi*t/dt)
     
     #Flux total entrant
     F_entrant = F_ray + F_cond + Flux_soleil
     
     #Energie par unite de surface
     E = F_entrant * dt
     
     
     return E



