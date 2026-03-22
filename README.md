# Analyse d'IRM

Ce projet de recherche vise à automatiser la segmentation du tronc cérébral à partir d'images IRM 3D (pondérées T1), une tâche cruciale pour le ciblage thérapeutique en neurochirurgie et radiothérapie. L'objectif était de surpasser les méthodes de segmentation multi-atlas traditionnelles en utilisant des réseaux de neurones convolutifs profonds.

Le pipeline technique développé comprend :
- Architecture Deep Learning : Implémentation d'un modèle U-Net 3D optimisé, capable de capturer des caractéristiques spatiales complexes tout en préservant la résolution locale via des connexions sautées (skip connections).
- Pipeline de Prétraitement : Standardisation des données (NIfTI), rééchantillonnage à une résolution isotrope de 1mm et normalisation d'intensité pour assurer la convergence du modèle.
- Data Augmentation & Régularisation : Application de transformations élastiques et de rotations pour pallier la taille limitée du dataset et prévenir le surapprentissage.
- Métriques de Performance : Évaluation rigoureuse basée sur le coefficient de Dice et la distance de Hausdorff pour mesurer la précision des contours anatomiques.

Résultats clés :
- Le modèle a atteint un score de Dice moyen de 0,91 sur l'ensemble de test, démontrant une haute fidélité par rapport aux segmentations manuelles des experts.
- Réduction significative du temps de traitement par rapport aux méthodes atlas classiques, permettant une analyse en quasi temps réel.
- Analyse approfondie de l'influence de la fonction de perte (Loss function) et du taux d'apprentissage sur la stabilité du réseau.
