import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, save
from scipy.ndimage import gaussian_filter
from dipy.segment.mask import median_otsu

def get_grads_from_txt(txt_path):
    """Lit le fichier gradient_directions_b-values.txt"""
    try:
        table = np.loadtxt(txt_path)
    except ValueError:
        table = np.genfromtxt(txt_path)
    bvecs = table[:, 0:3]
    bvals = table[:, 3]
    return bvals, bvecs

def preprocess_data(data, affine):
    """
    1. Calcule un masque propre (Median Otsu).
    2. Applique un lissage Gaussien pour réduire le 'bruit de neige'.
    """
    print("--- Pré-traitement (Masque + Lissage) ---")
    
    # 1. Masque automatique robuste (Otsu)
    # median_radius=2 nettoie les petits points blancs isolés
    # numpass=1 suffit généralement
    print("  -> Calcul du masque (Otsu)...")
    b0_data, mask = median_otsu(data, vol_idx=[0], median_radius=2, numpass=1)
    
    # 2. Lissage Gaussien (Smoothing)
    # On lisse un peu (sigma=0.75) en x,y,z mais PAS en temporel (0)
    print("  -> Lissage Gaussien (Denoising)...")
    # sigma = [x, y, z, t]
    data_smoothed = gaussian_filter(data, sigma=[0.75, 0.75, 0.75, 0])
    
    # On remet le masque propre par dessus pour nettoyer les bords flous
    data_smoothed[~mask] = 0
    
    return data_smoothed, mask

def fit_tensor_ols(data, bvals, bvecs, mask):
    print("--- 1. Estimation du tenseur (OLS) ---")
    S = data.copy()
    S[S <= 0] = 1e-6 
    
    log_S = np.log(S)
    log_S = np.nan_to_num(log_S) 
    
    num_vols = len(bvals)
    X = np.zeros((num_vols, 7))
    
    for i in range(num_vols):
        b = bvals[i]
        gx, gy, gz = bvecs[i]
        X[i, :] = [1.0, -b*gx**2, -b*gy**2, -b*gz**2, -2*b*gx*gy, -2*b*gx*gz, -2*b*gy*gz]

    try:
        X_pinv = np.linalg.pinv(X)
    except np.linalg.LinAlgError:
        return None

    n_x, n_y, n_z, n_t = data.shape
    mask_indices = np.where(mask)
    S_masked = log_S[mask_indices]
    
    beta = np.dot(S_masked, X_pinv.T)
    
    tensor_vol = np.zeros((n_x, n_y, n_z, 6), dtype=np.float32)
    # Mapping [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
    mapping = [1, 4, 2, 5, 6, 3]
    for k in range(6):
        tensor_vol[mask_indices[0], mask_indices[1], mask_indices[2], k] = beta[:, mapping[k]]

    return tensor_vol

def compute_maps_and_peaks(tensor_data, mask):
    print("--- 2. Calcul FA, ADC, RGB ---")
    nx, ny, nz, _ = tensor_data.shape
    
    FA_map = np.zeros((nx, ny, nz), dtype=np.float32)
    ADC_map = np.zeros((nx, ny, nz), dtype=np.float32)
    RGB_map = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    Peaks_map = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    
    ix, iy, iz = np.where(mask)
    
    for i, j, k in zip(ix, iy, iz):
        t = tensor_data[i, j, k]
        D_mat = np.array([[t[0], t[1], t[3]], [t[1], t[2], t[4]], [t[3], t[4], t[5]]])
        
        try:
            evals, evecs = np.linalg.eigh(D_mat)
        except: continue
            
        sort = np.argsort(evals)[::-1]
        evals = evals[sort]
        evecs = evecs[:, sort]
        
        # On force les valeurs propres positives (physiquement obligatoires)
        l1 = max(1e-6, evals[0])
        l2 = max(1e-6, evals[1])
        l3 = max(1e-6, evals[2])
        
        ADC_map[i, j, k] = (l1 + l2 + l3) / 3.0
        
        denom = np.sqrt(l1**2 + l2**2 + l3**2)
        if denom > 0:
            numer = np.sqrt((l1-l2)**2 + (l2-l3)**2 + (l1-l3)**2)
            fa = np.sqrt(0.5) * (numer / denom)
            FA_map[i, j, k] = min(max(fa, 0), 1)
            
            v1 = evecs[:, 0]
            Peaks_map[i, j, k] = v1
            # Couleur rehaussée pour meilleure visibilité
            RGB_map[i, j, k] = (FA_map[i, j, k] * np.abs(v1))
            
    return FA_map, ADC_map, RGB_map, Peaks_map

def tracking_euclidien_random(FA, Peaks, affine, n_seeds=50000):
    print(f"--- 3. Tractographie ({n_seeds} seeds) ---")
    streamlines = []
    
    fa_thresh = 0.2
    # Angle limite : cos(60 degrés) = 0.5. Si produit scalaire < 0.5, virage trop brusque -> stop.
    angle_thresh = 0.5 
    
    wm_voxels = np.argwhere(FA > fa_thresh)
    if len(wm_voxels) == 0: return []

    indices = np.random.choice(len(wm_voxels), size=n_seeds)
    step = 0.5
    max_steps = 300 # Un peu plus long pour les grandes fibres
    
    for idx in indices:
        vox = wm_voxels[idx]
        seed = vox.astype(float) + np.random.rand(3)
        
        px, py, pz = int(seed[0]), int(seed[1]), int(seed[2])
        d0 = Peaks[px, py, pz]
        if np.linalg.norm(d0) < 0.1: continue
            
        pts = []
        # On sauvegarde le point central en world coordinates
        pts.append(np.dot(affine, np.append(seed, 1))[:3])
        
        for polarity in [1, -1]:
            curr_pos = seed.copy()
            curr_dir = d0 * polarity
            path = []
            
            for _ in range(max_steps):
                curr_pos += curr_dir * step
                cx, cy, cz = int(curr_pos[0]), int(curr_pos[1]), int(curr_pos[2])
                
                if (cx < 0 or cx >= FA.shape[0] or cy < 0 or cy >= FA.shape[1] or cz < 0 or cz >= FA.shape[2]): break
                
                # Critère 1: FA
                if FA[cx, cy, cz] < fa_thresh: break
                
                next_dir = Peaks[cx, cy, cz]
                
                # Alignement de direction
                dot_prod = np.dot(curr_dir, next_dir)
                if dot_prod < 0:
                    next_dir = -next_dir
                    dot_prod = -dot_prod
                
                # Critère 2: Courbure (Angle)
                # Si la fibre tourne trop fort, c'est probablement une erreur
                if dot_prod < angle_thresh: 
                    break
                    
                curr_dir = next_dir
                path.append(np.dot(affine, np.append(curr_pos, 1))[:3])
            
            if polarity == 1: pts.extend(path)
            else: pts = path[::-1] + pts
                
        if len(pts) > 10: # On garde seulement si la fibre a une longueur décente
            streamlines.append(np.array(pts))

    return streamlines

if __name__ == "__main__":
    try:
        img = nib.load('dmri.nii.gz')
        raw_data = img.get_fdata()
        affine = img.affine
        header = img.header
    except:
        print("Erreur: dmri.nii introuvable.")
        exit()

    bvals, bvecs = get_grads_from_txt('gradient_directions_b-values.txt')

    # --- AMELIORATION 1 : PRE-TRAITEMENT ---
    data, mask = preprocess_data(raw_data, affine)
    
    # On sauvegarde le masque utilisé pour vérification
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, header), 'mask_used.nii.gz')

    # Fit
    tensor_img = fit_tensor_ols(data, bvals, bvecs, mask)
    
    if tensor_img is not None:
        t5d = tensor_img.reshape(tensor_img.shape[0], tensor_img.shape[1], tensor_img.shape[2], 1, 6)
        nib.save(nib.Nifti1Image(t5d, affine, header), 'tensors_ants_format.nii.gz')
        
        FA, ADC, RGB, Peaks = compute_maps_and_peaks(tensor_img, mask)
        
        nib.save(nib.Nifti1Image(FA, affine, header), 'FA.nii.gz')
        nib.save(nib.Nifti1Image(ADC, affine, header), 'ADC.nii.gz')
        nib.save(nib.Nifti1Image(RGB, affine, header), 'RGB.nii.gz')
        nib.save(nib.Nifti1Image(Peaks, affine, header), 'peaks.nii.gz')
        
        # --- AMELIORATION 2 : TRACKING PLUS STRICT ---
        streams = tracking_euclidien_random(FA, Peaks, affine, n_seeds=50000)
        
        if len(streams) > 0:
            print(f"Fibres générées : {len(streams)}")
            tractogram = Tractogram(streams, affine_to_rasmm=np.eye(4))
            save(tractogram, "white_matter_random.trk", header=header)
        else:
            print("Aucune fibre.")
            
    print("Terminé.")