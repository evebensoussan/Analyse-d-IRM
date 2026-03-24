import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def afficher_3_vues(img, tranche_axial=None, tranche_coronal=None, tranche_sagittal=None):
    """
    Affiche simultanément les 3 plans orthogonaux (axial, coronal, sagittal)
    d’une image NIfTI, avec des curseurs (sliders) permettant de naviguer dans les coupes.
    L’affichage respecte les proportions physiques du volume (espacements voxel).
    """
    # Récupération des données d’intensité sous forme de tableau NumPy
    data = img.get_fdata()
    affine = img.affine

    # Si l’image est 4D (par exemple une séquence temporelle), on garde uniquement la première composante
    if data.ndim == 4:
        data = data[..., 0]

    nx, ny, nz = data.shape  # Dimensions du volume

    # Récupération des espacements voxel (en mm)
    zooms = img.header.get_zooms()[:3]  # Exemple : (1.0, 1.0, 1.2)
    sx, sy, sz = zooms

    # Définition des tranches initiales (au centre du volume si non précisées)
    if tranche_axial is None:
        tranche_axial = nz // 2
    if tranche_coronal is None:
        tranche_coronal = ny // 2
    if tranche_sagittal is None:
        tranche_sagittal = nx // 2

    # --- Création de la figure avec 3 sous-graphiques ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    # --- Affichage des 3 vues initiales ---
    # Vue axiale (plan z)
    im_axial = axes[0].imshow(
        data[:, :, tranche_axial].T,
        cmap="gray", origin="lower",
        extent=[0, nx * sx, 0, ny * sy]
    )
    axes[0].set_title(f"Axiale (z={tranche_axial})")

    # Vue coronale (plan y)
    im_coronal = axes[1].imshow(
        data[:, tranche_coronal, :].T,
        cmap="gray", origin="lower",
        extent=[0, nx * sx, 0, nz * sz]
    )
    axes[1].set_title(f"Coronale (y={tranche_coronal})")

    # Vue sagittale (plan x)
    im_sagittal = axes[2].imshow(
        data[tranche_sagittal, :, :].T,
        cmap="gray", origin="lower",
        extent=[0, ny * sy, 0, nz * sz]
    )
    axes[2].set_title(f"Sagittale (x={tranche_sagittal})")

    # --- Création des sliders pour naviguer dans les coupes ---
    ax_slider_axial = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_slider_coronal = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_slider_sagittal = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")

    slider_axial = Slider(ax_slider_axial, "Axial", 0, nz - 1, valinit=tranche_axial, valfmt="%0.0f")
    slider_coronal = Slider(ax_slider_coronal, "Coronal", 0, ny - 1, valinit=tranche_coronal, valfmt="%0.0f")
    slider_sagittal = Slider(ax_slider_sagittal, "Sagittal", 0, nx - 1, valinit=tranche_sagittal, valfmt="%0.0f")

    # --- Fonctions de mise à jour lors du déplacement des sliders ---
    def update_axial(val):
        idx = int(slider_axial.val)
        im_axial.set_data(data[:, :, idx].T)
        axes[0].set_title(f"Axiale (z={idx})")
        fig.canvas.draw_idle()

    def update_coronal(val):
        idx = int(slider_coronal.val)
        im_coronal.set_data(data[:, idx, :].T)
        axes[1].set_title(f"Coronale (y={idx})")
        fig.canvas.draw_idle()

    def update_sagittal(val):
        idx = int(slider_sagittal.val)
        im_sagittal.set_data(data[idx, :, :].T)
        axes[2].set_title(f"Sagittale (x={idx})")
        fig.canvas.draw_idle()

    # Liaison des sliders à leurs fonctions respectives
    slider_axial.on_changed(update_axial)
    slider_coronal.on_changed(update_coronal)
    slider_sagittal.on_changed(update_sagittal)

    # Affichage final de la fenêtre interactive
    plt.show()


def main():
    # Gestion des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Visualiseur interactif NIfTI avec 3 vues orthogonales et sliders")
    parser.add_argument("image", type=str, help="Fichier NIfTI (.nii ou .nii.gz) à visualiser")
    args = parser.parse_args()

    # Chargement du volume et affichage des vues
    img = nib.load(args.image)
    afficher_3_vues(img)


# Point d’entrée du programme
if __name__ == "__main__":
    main()

# Ce script permet de visualiser une image médicale au format NIfTI selon les trois plans anatomiques
# (axial, coronal, sagittal). Grâce aux sliders interactifs, il est possible de naviguer librement
# dans les coupes du volume pour explorer la structure interne. Les proportions de chaque vue sont
# ajustées selon les espacements voxel pour garantir un affichage fidèle à l’échelle réelle.
# Outil très utile pour explorer visuellement un volume IRM ou scanner avant analyse.
