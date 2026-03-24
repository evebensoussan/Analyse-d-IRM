#Eve BEN SOUSSAN	Cip: bene2060			IMN708
#Léhna BOUCHAMA Cip: boul5260

import numpy as np # pour les calculs numériques et manipulations de tableaux
import imageio.v3 as iio # pour lire les images depuis des fichiers
import sys  # pour récupérer les arguments passés au script

# Fonction SSD (somme des différences au carré)
def ssd(I, J):
    """Somme des différences au carré."""
    I = I.astype(float)   # conversion en float pour précision
    J = J.astype(float)
    return np.sum((I - J) ** 2) # somme des carrés des différences pixel à pixel

# Fonction CR (coefficient de corrélation)
def cr(I, J):
    """Coefficient de corrélation."""
    I = I.astype(float)
    J = J.astype(float)
    mean_I = np.mean(I)  # moyenne des pixels de I
    mean_J = np.mean(J)  # moyenne des pixels de J

    numerator = np.sum((I - mean_I) * (J - mean_J))  # covariance
    denominator = np.sqrt(np.sum((I - mean_I)**2) * np.sum((J - mean_J)**2))
    if denominator == 0:   # éviter division par zéro
        return 0
    return numerator / denominator

# Fonction joint_hist (histogramme conjoint normalisé)

def joint_hist(I, J, bins=64):
    """Histogramme conjoint normalisé."""
    I = I.astype(float)
    J = J.astype(float)
     # normalisation des valeurs dans [0, 1]
    I = (I - I.min()) / (I.max() - I.min())
    J = (J - J.min()) / (J.max() - J.min())

# initialisation de l'histogramme
    H = np.zeros((bins, bins))

    # conversion des valeurs normalisées en indices
    idx_I = np.floor(I * (bins - 1)).astype(int)
    idx_J = np.floor(J * (bins - 1)).astype(int)

    # remplissage de l'histogramme conjoint
    for k in range(I.size):
        H[idx_I.flat[k], idx_J.flat[k]] += 1
    # normalisation pour obtenir une distribution de probabilité
    H /= np.sum(H)
    return H

# Fonction IM (information mutuelle)
def IM(I, J, bins=64):
    """Information mutuelle entre deux images."""
    H = joint_hist(I, J, bins)  # histogramme conjoint normalisé
    pI = np.sum(H, axis=1)  # probabilité marginale de I
    pJ = np.sum(H, axis=0)    # probabilité marginale de J

    eps = 1e-12   # petit epsilon pour éviter log(0)
    H = np.maximum(H, eps)
    pI = np.maximum(pI, eps)
    pJ = np.maximum(pJ, eps)

# calcul de l'information mutuelle
    return np.sum(H * np.log(H / (pI[:, None] * pJ[None, :])))

# Fonction principale

def main():
    # vérification des arguments passés au script
    if len(sys.argv) < 3:
        print("Utilisation : python compare_pairs.py img1.jpg img2.jpg [bins]")
        sys.exit(1)
# lecture des chemins des images et du nombre de bins
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    bins = int(sys.argv[3]) if len(sys.argv) > 3 else 64
# lecture des images
    I = iio.imread(img1_path)
    J = iio.imread(img2_path)/
# conversion en niveaux de gris si images RGB
    if I.ndim == 3:
        I = np.mean(I, axis=2)
    if J.ndim == 3:
        J = np.mean(J, axis=2)
# vérification que les images ont la même taille
    if I.shape != J.shape:
        raise ValueError("Les deux images doivent avoir la même taille.")
 # affichage des résultats
    print("\n Comparaison des images :")
    print(f" - SSD (somme des différences au carré) : {ssd(I, J):.2f}")
    print(f" - Corrélation (CR) : {cr(I, J):.4f}")
    print(f" - Information mutuelle (IM) : {IM(I, J, bins):.4f}")
    print(" Analyse terminée.\n")

# Lancement du script

if __name__ == "__main__":
    main()
