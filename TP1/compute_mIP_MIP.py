import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def compute_projection(data, axis, mode):
    """
    Calcule une projection maximale (MIP) ou minimale (mIP) le long d’un axe donné.
    - data : tableau NumPy contenant les intensités du volume.
    - axis : axe selon lequel effectuer la projection (0=sagittal, 1=coronal, 2=axial).
    - mode : 'max' pour une MIP ou 'min' pour une mIP.
    """
    if mode == "max":
        return np.max(data, axis=axis)
    elif mode == "min":
        return np.min(data, axis=axis)
    else:
        raise ValueError("mode doit être 'max' ou 'min'")

def main():
    # --- Définition des arguments du script ---
    parser = argparse.ArgumentParser(description="Calcul d’une projection MIP/mIP pour une image NIfTI")
    parser.add_argument("image", type=str, help="Fichier NIfTI (.nii ou .nii.gz) à traiter")
    parser.add_argument("--axis", type=int, default=2, choices=[0,1,2],
                        help="Axe de projection : 0 = sagittal, 1 = coronal, 2 = axial (par défaut)")
    parser.add_argument("--mode", type=str, default="max", choices=["max","min"],
                        help="Type de projection : 'max' pour MIP ou 'min' pour mIP")
    args = parser.parse_args()

    # --- Chargement de l’image NIfTI ---
    img = nib.load(args.image)
    data = img.get_fdata()                # Extraction des données sous forme de tableau NumPy
    zooms = img.header.get_zooms()[:3]    # Récupération des espacements entre voxels (en mm)

    # Si l’image est 4D (par exemple une série temporelle), on prend uniquement la première composante
    if data.ndim == 4:
        data = data[..., 0]

    axis = args.axis
    mode = args.mode

    # --- Calcul de la projection ---
    proj = compute_projection(data, axis, mode)

    # --- Calcul de l’étendue spatiale pour un affichage respectant les proportions réelles ---
    if axis == 0:  # Plan sagittal
        extent = [0, proj.shape[1]*zooms[2], 0, proj.shape[0]*zooms[1]]
    elif axis == 1:  # Plan coronal
        extent = [0, proj.shape[1]*zooms[2], 0, proj.shape[0]*zooms[0]]
    else:  # Plan axial
        extent = [0, proj.shape[1]*zooms[1], 0, proj.shape[0]*zooms[0]]

    # --- Affichage du résultat ---
    plt.figure(figsize=(6,6))
    plt.imshow(proj.T, cmap="gray", origin="lower", extent=extent, aspect="auto")
    plt.title(f"{mode.upper()} projection (axe {axis})")
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()

# Point d’entrée du script
if __name__ == "__main__":
    main()

# Ce script permet de calculer et d’afficher une projection maximale (MIP) ou minimale (mIP)
# d’une image médicale 3D (ou 4D) au format NIfTI. Les MIP/mIP sont souvent utilisées en imagerie
# médicale pour visualiser les structures les plus intenses (ou les moins intenses) le long d’un axe,
# par exemple pour observer les vaisseaux en angiographie ou évaluer la distribution du signal IRM.
# L’utilisateur peut choisir l’axe de projection (sagittal, coronal, axial) et le type de projection.
