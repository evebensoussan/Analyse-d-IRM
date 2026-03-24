#imports
import nibabel as nib
import numpy as np
import os
import pydicom
import sys
import re
import imageio.v2 as imageio

# ----------------------------------------------------------
# 1. Lecture du nom du fichier en argument
# ----------------------------------------------------------
if len(sys.argv) < 2:
    print(" Erreur : Aucun fichier fourni.")
    print(" Utilisation : python lire_image.py <nom_du_fichier>")
    print(" Exemple : python lire_image.py irm.nii.gz")
    sys.exit(1)

filename = sys.argv[1]

if not os.path.exists(filename):
    print(f" Erreur : le fichier '{filename}' n'existe pas dans le dossier courant ({os.getcwd()})")
    sys.exit(1)

# ----------------------------------------------------------
# 2. Détection du format du fichier
# ----------------------------------------------------------
ext = os.path.splitext(filename)[1].lower()
print(f"\n Fichier détecté : {filename}")
print(f"   Extension : {ext}")

# ----------------------------------------------------------
# 3. Lecture et analyse selon le type
# ----------------------------------------------------------
if ext in [".nii", ".gz"]:
    print("\n Lecture d'une image NIfTI / Analyze avec nibabel...")

    try:
        # ----------------------------------------------------------
        # Infos générales
        # ----------------------------------------------------------
        
        # Récupération des métadonnées associées à l’image
        img = nib.load(filename)
        hdr = img.header
        data = img.get_fdata()
        affine = img.affine

        # Informations de base
        print("\n=== Informations générales ===")
        print(f"Taille de l'image (en voxels) : {img.shape}")

        # Taille d'un voxel
        voxel_sizes = hdr.get_zooms()[:3]
        print(f"\nTaille d’un voxel : {voxel_sizes}")

        # Taille physique totale
        physical_size = tuple(np.array(img.shape[:3]) * np.array(voxel_sizes))
        print(f"\nTaille totale de l'image : {physical_size} ")

        print(f"\nMatrice affine :{affine}")

        print(f"\nType de données : {data.dtype}")

        # ----------------------------------------------------------
        # Statistiques et contrastes
        # ----------------------------------------------------------
        Imax = np.max(data)
        Imin = np.min(data)
        mean_intensity = np.mean(data)
        std_intensity = np.std(data)

        # ----------------------------------------------------------
        # Contrastes
        # ----------------------------------------------------------
        
        contrast_michelson = (Imax - Imin) / (Imax + Imin) if (Imax + Imin) != 0 else np.nan
        contrast_rms = std_intensity / mean_intensity if mean_intensity != 0 else np.nan

        print("\n=== Contrastes ===")
        print(f"\nContraste de Michelson : {contrast_michelson}")
        print(f"\nContraste RMS : {contrast_rms}")


    except Exception as e:
        print(f"Erreur lors de la lecture du fichier NIfTI/Analyze : {e}")

# ----------------------------------------------------------
# 4. Lecture d'un fichier DICOM
# ----------------------------------------------------------
elif ext in [".dcm", ""]:
    print("\n Lecture d’un fichier DICOM avec pydicom...")

    try:

        # ----------------------------------------------------------
        # Infos générales
        # ----------------------------------------------------------

        dicom_data = pydicom.dcmread(filename)
        print("\n=== Informations générales ===")
        print(f"\nDimensions : {getattr(dicom_data, 'Rows', '?')} x {getattr(dicom_data, 'Columns', '?')}")
        print(f"\nPixel Spacing : {getattr(dicom_data, 'PixelSpacing', 'Non spécifié')}")

        # Récupération des pixels si dispo
        if hasattr(dicom_data, "pixel_array"):
            data = dicom_data.pixel_array.astype(float)
            print(f"\nImage chargée : {data.shape}")
            print(f"\nType : {data.dtype}")
            print(f"\nValeurs min/max : {np.min(data)} / {np.max(data)}")

            # ----------------------------------------------------------
            # Contrastes
            # ----------------------------------------------------------

            Imax = np.max(data)
            Imin = np.min(data)
            mean_intensity = np.mean(data)
            std_intensity = np.std(data)

            contrast_michelson = (Imax - Imin) / (Imax + Imin) if (Imax + Imin) != 0 else np.nan
            contrast_rms = std_intensity / mean_intensity if mean_intensity != 0 else np.nan

            print("\n=== Contrastes ===")
            print(f"\nContraste de Michelson : {contrast_michelson}")
            print(f"\nContraste RMS : {contrast_rms}")

        else:
            print("Aucun pixel data trouvé dans le fichier DICOM.")

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier DICOM : {e}")


# ----------------------------------------------------------
# 4. Lecture d'un fichier VFF
# ----------------------------------------------------------
elif ext in [".vff"]:
    print("\n Lecture d’un fichier VFF avec pydicom...")

    try:
        print("\nLecture d'un fichier VFF (Visualization File Format)...")

        if len(sys.argv) < 2:
            print("Usage: python show_vff_stats.py <fichier.vff>")
            sys.exit(1)

        filename = sys.argv[1]

        if not os.path.exists(filename):
            print(f"Le fichier '{filename}' n'existe pas")
            sys.exit(1)

        with open(filename, "rb") as f:
            content = f.read()

        # --- Repérer fin du header VFF ---
        header_end = content.find(b'format=slice;')
        if header_end == -1:
            raise ValueError("Impossible de trouver 'format=slice;' dans le header VFF")

        # Lire header complet ASCII
        i = header_end + 13
        while i < len(content) and (32 <= content[i] <= 126 or content[i] in (9,10,13)):
            i += 1

        header_bytes = content[:i]
        raw_data = content[i:]

        # --- Extraire dimensions et bits ---
        match = re.search(rb"size=(\d+)\s+(\d+)\s+(\d+)", header_bytes)
        if not match:
            raise ValueError("Impossible de lire la taille dans le header VFF")
        nx, ny, nz = map(int, match.groups())
        expected_size = nx * ny * nz
        print(f"Dimensions détectées : nx={nx}, ny={ny}, nz={nz}")

        match_bits = re.search(rb"bits=(\d+)", header_bytes)
        bits = int(match_bits.group(1)) if match_bits else 16
        print(f"Bits par voxel : {bits}")

        dtype = np.int16 if bits==16 else np.float32

        # --- Taille des voxels ---
        match_vox = re.search(rb"(?:spacing|voxel)=(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", header_bytes)
        if match_vox:
            vx, vy, vz = map(float, match_vox.groups())
            print(f"Taille des voxels : vx={vx} , vy={vy} , vz={vz} ")
            taille_x = nx * vx
            taille_y = ny * vy
            taille_z = nz * vz
            print(f"Taille totale de l'image : {taille_x:.2f}  × {taille_y:.2f}  × {taille_z:.2f} ")
        else:
            print(" Aucune information de taille de voxel trouvée dans le header.")
            vx = vy = vz = np.nan

        # --- Ajuster raw_data pour être multiple de dtype ---
        remainder = len(raw_data) % np.dtype(dtype).itemsize
        if remainder != 0:
            raw_data = raw_data[:-remainder]

        # --- Lire les données ---
        data = np.frombuffer(raw_data, dtype=dtype)

        # Ajustement taille
        if data.size > expected_size:
            data = data[:expected_size]
        elif data.size < expected_size:
            padding = np.zeros(expected_size - data.size, dtype=dtype)
            data = np.concatenate([data, padding])

        # Reshape en 3D
        data = data.reshape((nz, ny, nx)).astype(float)

        # --- Statistiques ---
        Imin = np.min(data)
        Imax = np.max(data)
        mean_intensity = np.mean(data)
        std_intensity = np.std(data)

        contrast_michelson = (Imax - Imin)/(Imax + Imin) if (Imax + Imin)!=0 else np.nan
        contrast_rms = std_intensity/mean_intensity if mean_intensity!=0 else np.nan


        print("\n=== Contrastes ===")
        print(f"Contraste de Michelson : {contrast_michelson:.3f}")
        print(f"Contraste RMS : {contrast_rms:.3f}")

        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier VFF : {e}")

# ----------------------------------------------------------
# 4. Lecture d'un fichier PGM
# ----------------------------------------------------------
elif ext in [".pgm"]:
    print("\n Lecture d’un fichier PGM avec pydicom...")

    try:
        # --- Lecture du fichier PGM ---
        with open(filename, "rb") as f:
            header = f.readline().strip()  # Ex : b'P5'
            if header not in [b'P2', b'P5']:
                raise ValueError("Format PGM non reconnu (doit être P2 ou P5)")

            # Lecture des métadonnées
            while True:
                line = f.readline()
                if not line.startswith(b'#'):  # Sauter les commentaires
                    width, height = map(int, line.split())
                    break

            maxval = int(f.readline().strip())
            data = imageio.imread(filename).astype(float)

        print(f"Image PGM chargée : {data.shape}, Type : {data.dtype}")
        print(f"Taille de l’image (voxels) : largeur={width}, hauteur={height}")
        print(f"Valeur maximale du voxels (échelle) : {maxval}")


        # Statistiques
        Imin = np.min(data)
        Imax = np.max(data)
        mean_intensity = np.mean(data)
        std_intensity = np.std(data)

        contrast_michelson = (Imax - Imin)/(Imax + Imin) if (Imax + Imin)!=0 else np.nan
        contrast_rms = std_intensity/mean_intensity if mean_intensity!=0 else np.nan


        print("\n=== Contrastes ===")
        print(f"Contraste de Michelson : {contrast_michelson:.3f}")
        print(f"Contraste RMS : {contrast_rms:.3f}")


    except Exception as e:
        print(f"Erreur lors de la lecture du fichier PGM : {e}")

# ----------------------------------------------------------
# 5. Format non reconnu
# ----------------------------------------------------------
else:
    print(f"Format de fichier non reconnu : {ext}")
    print(" Formats supportés : .nii, .nii.gz, .dcm, .vff, .pgm")

