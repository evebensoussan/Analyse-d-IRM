import nibabel as nib
import nibabel.orientations as nio

# Importation des bibliothèques nécessaires :
# - nibabel : pour charger et manipuler les images médicales au format NIfTI (.nii)
# - nibabel.orientations : pour interroger l’orientation spatiale des axes (RAS, LPI, etc.)

# Chargement de l'image NIfTI FLAIR
img = nib.load("flair.nii")

# Affichage de la matrice affine associée au volume
# La matrice affine permet de faire le lien entre les indices du tableau (i, j, k)
# et les coordonnées réelles dans l’espace du patient (en millimètres).
print("Affine matrix:\n", img.affine)

# Détermination de l’orientation des axes selon la convention médicale
# (par exemple RAS = Right-Anterior-Superior ou LPI = Left-Posterior-Inferior)
orientation = nio.aff2axcodes(img.affine)
print("Orientation des axes :", orientation)

# Chargement des données en mémoire sous forme de tableau NumPy
data = img.get_fdata()

# Vérification de l’orientation après chargement
# (elle reste la même que celle déterminée à partir de la matrice affine)
print("Orientation après chargement :", nio.aff2axcodes(img.affine))

# Ce script permet de vérifier la cohérence de l’orientation spatiale de l’image IRM FLAIR.
# Il affiche la matrice affine (liant les voxels à l’espace patient) ainsi que le code d’orientation
# des axes, utile pour s’assurer que le volume est correctement aligné avant tout traitement ou visualisation.
