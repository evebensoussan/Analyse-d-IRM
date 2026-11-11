# 📦 Données

Les données d'imagerie ne sont **pas versionnées** dans ce dépôt :

- elles sont volumineuses (les volumes CT et IRM dépassent les limites de GitHub : 100 Mo par fichier) ;
- elles proviennent du matériel pédagogique du cours IMN 530 / IMN 708 (Université de Sherbrooke).

Le dossier `data/` est ignoré par Git (voir `.gitignore`).

## Fichiers attendus

| Fichier | Utilisé dans | Description |
|---|---|---|
| `flair.nii`, `t1.nii` | TP1, TP2 | IRM cérébrales FLAIR et T1 |
| `epaule.pgm`, `rat111.vff`, `slice06.nii`, `us.nii` | TP1 | Exemples multimodalités (CT, µCT, IRM cardiaque, ultrason) |
| `I1.png … J6.jpg`, `BrainMRI_1…4.jpg` | TP2 | Paires d'images pour les critères de similarité et le recalage 2D |
| `NIFTI/T1w.nii.gz`, `atlases/` | TP2 (ANTs) | T1, atlas MNI, JHU-ICBM, thalamus |
| `fmri.nii`, `ideal.txt` | TP3 | IRMf (tâche « Roland ») et régresseur déjà convolué avec la HRF |
| `dmri.nii`, `gradient_directions_b-values.txt` | TP3 | IRM de diffusion, b-vecteurs et b-valeurs |

## Pour reproduire

1. Récupérer les données auprès du cours (site du cours ou enseignant).
2. Les placer dans `data/` en respectant les noms ci-dessus.
3. Lancer les scripts depuis la racine du dépôt.
