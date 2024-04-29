import matplotlib.pyplot as plt


# Fonction qui trace toutes distribution de temperature
def plot_temperature (x, z, Tr, label):
    #plt.pcolor(x, z, Tr, cmap='hot', vmin=270, vmax=340) #Certaines variations de T sont trop petites pour etre remarquables en couleur avec ces parametres
    plt.pcolor(x, z, Tr, vmin=270, vmax=340)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title('T(x,z) [K]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().invert_yaxis()
    plt.savefig(f'{label}.png')
    plt.clf()