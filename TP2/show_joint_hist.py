#Eve BEN SOUSSAN	Cip: bene2060			IMN708
#Léhna BOUCHAMA Cip: boul5260

import numpy as np # pour manipuler les tableaux et faire les calculs numériques
import matplotlib.pyplot as plt # pour afficher les résultats graphiques
import imageio.v3 as iio # pour lire les images depuis des fichiers
import sys # pour récupérer les arguments passés au script

# Fonction pour calculer l'histogramme conjoint de deux images
def joint_hist(I, J, bins=64):
    # Vérification que les images ont la même taille
    if I.shape != J.shape:
        raise ValueError("Les deux images doivent avoir la même taille.")

# Conversion en float pour éviter les erreurs de calcul
    I = I.astype(float)
    J = J.astype(float)

    # Normalisation des valeurs dans [0, 1]
    I = (I - I.min()) / (I.max() - I.min())
    J = (J - J.min()) / (J.max() - J.min())
 # Initialisation de l'histogramme 2D
    H = np.zeros((bins, bins), dtype=float)
 # Conversion des valeurs normalisées en indices pour l'histogramme
    idx_I = np.floor(I * (bins - 1)).astype(int)
    idx_J = np.floor(J * (bins - 1)).astype(int)
# Parcours de tous les pixels pour remplir l'histogramme
    for k in range(I.size):
        H[idx_I.flat[k], idx_J.flat[k]] += 1

# Vérification : la somme de H doit correspondre au nombre total de pixels
    total_pixels = I.size
    total_hist = np.sum(H)

    print(f"\n🧮 Vérification de cohérence :")
    print(f" - Somme(H) = {total_hist:.0f}")
    print(f" - n × p = {total_pixels}")
    if abs(total_hist - total_pixels) < 1e-6:
        print(" ✅ OK : la somme de l’histogramme correspond bien au nombre total de pixels.\n")
    else:
        print(" ⚠️ ATTENTION : incohérence dans le calcul de l’histogramme !\n")

    return H

# Fonction principale
def main():
    # Vérification des arguments passés au script
    if len(sys.argv) < 3:
        print("Utilisation : python show_joint_hist.py img1.jpg img2.jpg [bins]")
        sys.exit(1)
# Lecture des chemins d'images et du nombre de bins
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    bins = int(sys.argv[3]) if len(sys.argv) > 3 else 64
 # Lecture des images
    I = iio.imread(img1_path)
    J = iio.imread(img2_path)
# Conversion en niveaux de gris si les images sont en couleur
    if I.ndim == 3:
        I = np.mean(I, axis=2)
    if J.ndim == 3:
        J = np.mean(J, axis=2)
# Calcul de l'histogramme conjoint
    H = joint_hist(I, J, bins)
# Affichage de l'histogramme avec matplotlib
    plt.figure(figsize=(6, 5))
    plt.imshow(H, cmap="turbo", origin="lower") # couleur pour mieux visualiser
    plt.title("Histogramme conjoint", fontsize=13)
    plt.xlabel("Valeurs de l'image J")
    plt.ylabel("Valeurs de l'image I")
    plt.colorbar(label='Fréquence brute')
    plt.tight_layout()
    plt.show()
# Lancement du script

if __name__ == "__main__":
    main()
