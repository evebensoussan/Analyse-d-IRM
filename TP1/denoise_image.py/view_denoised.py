import nibabel as nib
import matplotlib.pyplot as plt

# Importation des bibliothèques nécessaires :
# - nibabel pour la lecture des fichiers d’imagerie médicale au format NIfTI (.nii)
# - matplotlib.pyplot pour l’affichage des images sous forme de coupes 2D

# Liste des fichiers à afficher, correspondant aux différentes méthodes de filtrage appliquées sur les images FLAIR
files = [
    "results_flair/flair_median.nii",
    "results_flair/flair_bilateral.nii",
    "results_flair/flair_nlmeans.nii"
]

# Titres associés à chaque méthode pour l’affichage
titles = ["Median", "Bilateral", "NLMeans"]

# Boucle sur chaque fichier et son titre associé
for f, t in zip(files, titles):
    # Chargement du volume NIfTI
    img = nib.load(f)
    # Conversion du volume en tableau NumPy pour manipulation
    data = img.get_fdata()
    
    # Sélection de la coupe centrale selon l’axe z
    z = data.shape[2] // 2
    
    # Affichage de la coupe correspondante
    plt.figure()
    plt.imshow(data[:, :, z], cmap="gray", origin="lower")
    plt.title(t)           # Ajout du titre correspondant à la méthode de filtrage
    plt.axis("off")        # Suppression des axes pour une meilleure lisibilité
    plt.show()             # Affichage de la figure

# Ce script permet de visualiser et de comparer les effets des trois méthodes de filtrage (médian, bilatéral, NLMeans)
# sur les volumes IRM FLAIR. En affichant la même coupe centrale pour chaque image, on peut observer les différences
# de lissage, de préservation des contours et de réduction du bruit entre les approches.
