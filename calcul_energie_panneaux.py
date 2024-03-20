import numpy as np
import matplotlib.pyplot as plt

A_p= 1000 # Aire des panneaux solaires (m^2)
L=3.828*pow(10,26) # Luminosite du soleil (W)
d_TS=149597870700 # distance entre Planetes et Soleil PS (m)
d_MS=1.52*d_TS 
d_VS=0.72*d_TS
d_ES=5.2*d_TS
t=8760 # heaures dans une annee
t=12


eff = (15+23)/(2*100) # efficacite moyenne des panneaux solaires

def calcul_energie_panneaux(luminosite,distance, efficacite, aire, temps):
    I = luminosite/(4*np.pi*pow(distance,2)) # (W/m^2)
    print(I)
    I_eff = I*efficacite 
    P_eff = I_eff*aire # (W)
    E = P_eff*temps # (Wh), 1 Wh = 3600 J

    return E

calcul_energie_panneaux(L, d_TS, eff, A_p, t)
calcul_energie_panneaux(L, d_MS, eff, A_p, t)
calcul_energie_panneaux(L, d_VS, eff, A_p, t)
calcul_energie_panneaux(L, d_ES, eff, A_p, t)


#print(calcul_energie(L, d_TS, eff, A_p, t))
#print(calcul_energie(L, d_MS, eff, A_p, t))
#print(calcul_energie(L, d_VS, eff, A_p, t))
#print(calcul_energie(L, d_ES, eff, A_p, t))