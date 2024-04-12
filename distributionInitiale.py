import numpy as np    


# Définition de la distribution initiale
def distributionInitiale (Nx, Nz, Lx, Lz, Ts):
        # Création d'une grille Nx par Nz
        x = np.linspace(0, Lx, Nx)
        z = np.linspace(0, Lz, Nz)
        X, Z = np.meshgrid(x, z)

        # Paramètres de la gaussienne centrée
        sigma_x = Lx / 20  # Largeur de la gaussienne en x
        sigma_z = Lz / 20  # Largeur de la gaussienne en z
        mu_x = Lx / 2      # Position x du centre de la gaussienne
        mu_z = Lz / 2      # Position z du centre de la gaussienne

        # Calcul de la distribution gaussienne 2D
        gauss_dist = np.exp(-((X-mu_x)**2 / (2 * sigma_x**2) + (Z-mu_z)**2 / (2 * sigma_z**2)))

        # Ajustement pour le minimum et le maximum
        gauss_dist = Ts + (2 * Ts - Ts) * gauss_dist

        gauss_dist_col = np.ravel(gauss_dist, order='F')

        return gauss_dist_col