# Import des bibliothèques nécessaires
import argparse, os                   # argparse pour gérer les arguments de la ligne de commande, os pour gérer dossiers/fichiers
import numpy as np, nibabel as nib    # numpy pour les calculs, nibabel pour lire/écrire les fichiers NIfTI
from scipy.ndimage import median_filter  # Filtre médian 3D pour le débruitage
import matplotlib.pyplot as plt       # Pour l'affichage des images

# Fonction principale
def main():
    # ----------------------------------------------------------
    # 1. Gestion des arguments
    # ----------------------------------------------------------
    # Création d’un parser pour récupérer les arguments passés en ligne de commande
    # -i : fichier NIfTI d’entrée
    # -o : dossier de sortie
    # --size : taille du voisinage pour le filtre médian
    parser = argparse.ArgumentParser(description="Median denoising")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI")
    parser.add_argument("-o", "--outdir", default="results_median", help="Output folder")
    parser.add_argument("--size", type=int, default=3)  # Taille du filtre médian
    args = parser.parse_args()

    # Création du dossier de sortie si nécessaire
    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------
    # 2. Chargement de l’image NIfTI
    # ----------------------------------------------------------
    img = nib.load(args.input)                  # Lecture du fichier NIfTI
    data = img.get_fdata(dtype=np.float32)      # Conversion en float32 pour permettre un traitement précis

    # ----------------------------------------------------------
    # 3. Application du filtre médian 3D
    # ----------------------------------------------------------
    print(f"Running median filter (size={args.size})...")
    filtered = median_filter(data, size=(args.size,)*3)  
    # Le filtre médian remplace chaque voxel par la médiane de ses voisins dans un cube de taille "size"
    # Cela permet de réduire le bruit impulsionnel ou les pixels aberrants sans trop lisser les contours

    # ----------------------------------------------------------
    # 4. Sauvegarde du résultat
    # ----------------------------------------------------------
    out_path = os.path.join(args.outdir, "median.nii")  # Chemin de sortie
    nib.save(nib.Nifti1Image(filtered.astype(np.float32), img.affine), out_path)  # Sauvegarde en conservant la géométrie originale
    print(f"Saved: {out_path}")

    # ----------------------------------------------------------
    # 5. Affichage d’une coupe centrale
    # ----------------------------------------------------------
    mid_slice = filtered.shape[2] // 2  # Sélection de la coupe centrale
    plt.figure(figsize=(8, 6))
    plt.imshow(filtered[:, :, mid_slice].T, cmap='gray', origin='lower')  # Transposition pour une orientation correcte
    plt.title("Denoised (Median)")
    plt.axis('off')  # Supprime les axes pour un affichage plus clair
    plt.show()

# Exécution du script
if __name__ == "__main__":
    main()
