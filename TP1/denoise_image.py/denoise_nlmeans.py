# Import des bibliothèques nécessaires
import argparse, os                  # argparse pour gérer les arguments de la ligne de commande, os pour gérer les dossiers/fichiers
import numpy as np, nibabel as nib   # numpy pour les calculs, nibabel pour lire/écrire les fichiers NIfTI
from skimage.restoration import denoise_nl_means, estimate_sigma  # Fonctions de débruitage NLMeans et estimation du bruit
import matplotlib.pyplot as plt      # Pour l'affichage des images

# Fonction principale
def main():
    # ----------------------------------------------------------
    # 1. Gestion des arguments
    # ----------------------------------------------------------
    parser = argparse.ArgumentParser(description="Fast NLMeans denoising (2D slices)")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI file")   # Fichier NIfTI d'entrée
    parser.add_argument("-o", "--outdir", default="results_nlmeans", help="Output folder")  # Dossier de sortie
    parser.add_argument("--fast", action="store_true", help="Use fast (2D) mode")   # Option pour utiliser le mode 2D rapide
    args = parser.parse_args()  # Récupère les arguments passés par l'utilisateur

    # Création du dossier de sortie si nécessaire
    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------------------------------------
    # 2. Lecture de l'image NIfTI
    # ----------------------------------------------------------
    img = nib.load(args.input)                # Chargement de l'image NIfTI
    data = img.get_fdata(dtype=np.float32)    # Conversion en float32 pour les calculs

    # ----------------------------------------------------------
    # 3. Estimation du bruit
    # ----------------------------------------------------------
    sigma_est = np.mean(estimate_sigma(data, channel_axis=None))  # Estimation du sigma moyen sur l'image
    h = 0.8 * sigma_est   # Paramètre h pour NLMeans (influence du bruit)
    print(f"Estimated sigma={sigma_est:.4f}, using h={h:.4f}")

    # ----------------------------------------------------------
    # 4. Débruitage NLMeans
    # ----------------------------------------------------------
    if args.fast:
        # Mode rapide : traitement slice par slice en 2D
        print("Running fast 2D NLMeans slice-by-slice...")
        filtered = np.zeros_like(data)  # Initialisation du tableau filtré
        for i in range(data.shape[2]):  # On parcourt chaque coupe axiale
            filtered[:, :, i] = denoise_nl_means(
                data[:, :, i],
                h=h,
                patch_size=5,
                patch_distance=7,
                fast_mode=True
            )
    else:
        # Mode complet 3D (peut être très lent)
        print("Running full 3D NLMeans (can be very slow)...")
        filtered = denoise_nl_means(
            data,
            h=h,
            patch_size=5,
            patch_distance=7,
            fast_mode=False
        )

    # ----------------------------------------------------------
    # 5. Sauvegarde de l'image débruitée
    # ----------------------------------------------------------
    out_path = os.path.join(
        args.outdir,
        os.path.basename(args.input).replace(".nii", "_nlmeans.nii")
    )
    # Création et sauvegarde du fichier NIfTI avec les mêmes infos géométriques (affine)
    nib.save(nib.Nifti1Image(filtered.astype(np.float32), img.affine), out_path)
    print(f"Saved: {out_path}")

    # ----------------------------------------------------------
    # 6. Affichage d'une coupe centrale
    # ----------------------------------------------------------
    mid_slice = filtered.shape[2] // 2  # Coupe centrale
    plt.imshow(filtered[:, :, mid_slice].T, cmap='gray', origin='lower')  # Affichage avec transpose pour l'orientation
    plt.title("Denoised (NLMeans 2D)")
    plt.axis('off')  # Supprime les axes pour la visualisation
    plt.show()

# Exécution du script
if __name__ == "__main__":
    main()
