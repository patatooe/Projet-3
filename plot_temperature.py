import matplotlib.pyplot as plt


# Fonction qui trace toutes distribution de temperature
def plot_temperature (x, z, Tr, label):
    plt.pcolor(x, z, Tr, cmap='hot', vmin=280, vmax=320)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title('T$_0$(x,y) [K]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().invert_yaxis()
    plt.savefig(f'{label}.png')