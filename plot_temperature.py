import matplotlib.pyplot as plt


# Fonction qui trace toutes distribution de temperature
def plot_temperature (x, z, Tr, label):
    plt.pcolor(x, z, Tr, cmap='hot')#, vmin=240, vmax=310)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title('T(x,z) [K]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().invert_yaxis()
    plt.savefig(f'{label}.png')
    plt.clf()