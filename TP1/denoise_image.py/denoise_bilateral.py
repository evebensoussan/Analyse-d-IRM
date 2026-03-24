# Import des bibliothèques nécessaires
import argparse, os                  # argparse pour gérer les arguments de la ligne de commande, os pour gérer dossiers/fichiers
import numpy as np, nibabel as nib   # numpy pour les calculs, nibabel pour lire/écrire les fichiers NIfTI
from skimage.restoration import denoise_bilateral  # Filtre bilatéral pour le débruitage
import matplotlib.pyplot as plt      # Pour l'affichage des images

# Fonction principale
def main():
    # ----------------------------------------------------------
    # 1. Gestion des arguments
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(description="Bilateral denoising")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI")  # Fichier NIfTI d'entrée
    parser.add_argument("-o", "--outdir", default="results_bilateral", help="Output folder")  # Dossier de sortie
    parser.add_argument("--sigma_spatial", type=float, default=2.0)  # Paramètre spatial du filtre bilatéral
    parser.add_argument("--sigma_color", type=float, default=0.1)    # Paramètre d'intensité du filtre bilatéral
    args = parser.parse_args()  # Récupère les arguments passés par l'utilisateur

    # Création du dossier de sortie si nécessaire
    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------
    # 2. Lecture de l'image NIfTI
    # ----------------------------------------------------------
    img = nib.load(args.input)                    # Chargement du fichier NIfTI
    data = img.get_fdata(dtype=np.float32)        # Conversion en float32
    # Normalisation de l'image entre 0 et 1 pour que le filtre bilatéral fonctionne correctement
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))

    # ----------------------------------------------------------
    # 3. Débruitage slice par slice
    # ----------------------------------------------------------
    print(f"Running bilateral filter slice-by-slice...")
    filtered = np.zeros_like(norm)  # Initialisation du tableau filtré
    for i in range(norm.shape[2]):  # Parcours de chaque coupe axiale
        filtered[:,:,i] = denoise_bilateral(
            norm[:,:,i],
            sigma_color=args.sigma_color,    # Influence sur l'intensité
            sigma_spatial=args.sigma_spatial,  # Influence spatiale
            channel_axis=None
        )

    # Remise à l'échelle des valeurs pour retrouver l'intensité originale
    filtered = filtered * (np.max(data) - np.min(data)) + np.min(data)

    # ----------------------------------------------------------
    # 4. Sauvegarde de l'image filtrée
    # ----------------------------------------------------------
    out_path = os.path.join(args.outdir, "bilateral.nii")  # Chemin de sortie
    nib.save(nib.Nifti1Image(filtered.astype(np.float32), img.affine), out_path)  # Sauvegarde avec la même géométrie
    print(f"Saved: {out_path}")

    # ----------------------------------------------------------
    # 5. Affichage d'une coupe centrale
    # ----------------------------------------------------------
    mid_slice = filtered.shape[2] // 2  # Sélection de la coupe centrale
    plt.figure(figsize=(8, 6))
    plt.imshow(filtered[:, :, mid_slice].T, cmap='gray', origin='lower')  # Affichage avec transpose pour orientation correcte
    plt.title("Denoised (Bilateral)")
    plt.axis('off')  # Supprime les axes
    plt.show()

# Exécution du script
if __name__ == "__main__":
    main()
