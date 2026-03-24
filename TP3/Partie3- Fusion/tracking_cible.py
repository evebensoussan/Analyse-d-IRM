import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, save
from scipy.ndimage import label

# --- 1. LECTURE GRADIENTS ---
def get_grads_from_txt(txt_path):
    try:
        table = np.loadtxt(txt_path)
    except OSError:
        return None, None
    return table[:, 3], table[:, 0:3]

# --- 2. TENSEUR & FA ---
def fit_tensor_and_fa(data, bvals, bvecs, mask):
    print("--- Calculs Tenseur & FA ---")
    S = data.copy()
    S[S <= 0] = 1e-6
    log_S = np.log(S)
    
    X = np.zeros((len(bvals), 7))
    for i, b in enumerate(bvals):
        gx, gy, gz = bvecs[i]
        X[i, :] = [1.0, -b*gx**2, -b*gy**2, -b*gz**2, -2*b*gx*gy, -2*b*gx*gz, -2*b*gy*gz]

    try:
        X_pinv = np.linalg.pinv(X)
    except: return None, None

    mask_idx = np.where(mask)
    beta = np.dot(log_S[mask_idx], X_pinv.T)
    
    nx, ny, nz, _ = data.shape
    FA_map = np.zeros((nx, ny, nz), dtype=np.float32)
    Peaks_map = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    
    # Mapping indices tenseur
    mapping = [1, 4, 2, 5, 6, 3] 
    
    for idx, (i, j, k) in enumerate(zip(*mask_idx)):
        vec = beta[idx, :]
        D = np.zeros((3,3))
        D[0,0]=vec[1]; D[1,1]=vec[2]; D[2,2]=vec[3]
        D[0,1]=vec[4]; D[0,2]=vec[5]; D[1,2]=vec[6]
        D[1,0]=vec[4]; D[2,0]=vec[5]; D[2,1]=vec[6]
        
        try:
            evals, evecs = np.linalg.eigh(D)
            sort = np.argsort(evals)[::-1]
            l1, l2, l3 = evals[sort]
            l1=max(0,l1); l2=max(0,l2); l3=max(0,l3)
            
            if (l1+l2+l3) > 0:
                num = np.sqrt((l1-l2)**2 + (l2-l3)**2 + (l1-l3)**2)
                den = np.sqrt(l1**2 + l2**2 + l3**2)
                FA_map[i, j, k] = np.sqrt(0.5)*(num/den)
                Peaks_map[i, j, k] = evecs[:, sort[0]]
        except: pass
            
    return FA_map, Peaks_map

# --- 3. TRACTOGRAPHIE AVEC DEBUG ---
def tracking_smart_debug(FA, Peaks, affine, roi_path):
    print(f"--- Tractographie : Mode Nettoyage Avancé ---")
    
    try:
        roi_img = nib.load(roi_path)
        roi_data = np.squeeze(roi_img.get_fdata())
    except: return []

    # 1. ANALYSE DES ZONES
    labeled_array, num_features = label(roi_data > 0)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0 # Ignorer le fond
    
    # TRIER PAR TAILLE
    sorted_indices = np.argsort(sizes)[::-1] # Du plus grand au plus petit
    
    # --- FILTRE MAGIQUE ---
    # On garde max 4 zones, MAIS elles doivent faire au moins 50 pixels
    top_indices = []
    min_pixel_size = 50 
    
    print("   --- Analyse des zones trouvées ---")
    for i in range(min(num_features, 6)): # On regarde les 6 plus grosses
        idx = sorted_indices[i]
        taille = sizes[idx]
        msg = f"   Zone #{i+1} (ID {idx}) : {taille} voxels"
        
        if taille > min_pixel_size and len(top_indices) < 4:
            top_indices.append(idx)
            print(msg + " -> GARDÉE")
        else:
            print(msg + " -> REJETÉE (Trop petite ou limite atteinte)")

    # Création du masque propre
    mask_clean_bool = np.isin(labeled_array, top_indices)
    labeled_clean = np.where(mask_clean_bool, labeled_array, 0)
    
    # SAUVEGARDE DU FICHIER DEBUG (Pour vérifier dans MI-Brain)
    debug_img = nib.Nifti1Image(labeled_clean.astype(np.float32), affine)
    nib.save(debug_img, 'debug_zones_utilisees.nii.gz')
    print("   --> Fichier 'debug_zones_utilisees.nii.gz' créé pour vérification.")

    # --- REGLAGES ---
    seeds_per_voxel = 6
    seed_voxels = np.argwhere(mask_clean_bool & (FA > 0.15))
    
    streamlines = []
    inv_affine = np.linalg.inv(affine)
    step_size = 0.5
    
    # TRACTOGRAPHIE
    for idx in seed_voxels:
        for _ in range(seeds_per_voxel):
            seed_pos = idx.astype(float) + np.random.rand(3)
            px, py, pz = int(seed_pos[0]), int(seed_pos[1]), int(seed_pos[2])
            if np.linalg.norm(Peaks[px, py, pz]) == 0: continue

            full_streamline = []
            seed_world = np.dot(affine, np.append(seed_pos, 1))[:3]
            start_dir = Peaks[px, py, pz]

            for direction in [1, -1]:
                points = []
                if direction == 1: points.append(seed_world)
                curr_pos = seed_pos.copy()
                curr_dir = start_dir.copy() * direction
                
                for _ in range(300): 
                    curr_pos = curr_pos + curr_dir * step_size
                    cx, cy, cz = int(curr_pos[0]), int(curr_pos[1]), int(curr_pos[2])
                    
                    if (cx < 0 or cx >= FA.shape[0] or cy < 0 or cy >= FA.shape[1] or cz < 0 or cz >= FA.shape[2]): break
                    if FA[cx, cy, cz] < 0.15: break 
                    
                    new_dir = Peaks[cx, cy, cz]
                    if np.dot(curr_dir, new_dir) < 0: new_dir = -new_dir
                    if np.dot(curr_dir, new_dir) < 0.6: break
                    
                    curr_dir = new_dir
                    world_pos = np.dot(affine, np.append(curr_pos, 1))[:3]
                    points.append(world_pos)
                
                if direction == -1: full_streamline = points[::-1] + full_streamline
                else: full_streamline = full_streamline + points

            # FILTRE DE CONNEXION (Doit toucher au moins 2 zones valides)
            if len(full_streamline) > 30:
                stream_arr = np.array(full_streamline)
                ones = np.ones((len(stream_arr), 1))
                pts_vox = np.dot(inv_affine, np.hstack((stream_arr, ones)).T).T[:, :3].astype(int)
                
                zones_touchees = set()
                for pt in pts_vox:
                    if (0 <= pt[0] < labeled_clean.shape[0] and 
                        0 <= pt[1] < labeled_clean.shape[1] and 
                        0 <= pt[2] < labeled_clean.shape[2]):
                        val = labeled_clean[pt[0], pt[1], pt[2]]
                        if val > 0: zones_touchees.add(val)
                
                if len(zones_touchees) >= 2:
                    streamlines.append(stream_arr)

    return streamlines

# --- MAIN ---
if __name__ == "__main__":
    img = nib.load('dmri.nii.gz')
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    bvals, bvecs = get_grads_from_txt('gradient_directions_b-values.txt')
    mask = np.mean(data[..., bvals < 10], axis=3) > (np.mean(data) * 0.2)
    
    FA, Peaks = fit_tensor_and_fa(data, bvals, bvecs, mask)
    
    roi_file = 'activation_pour_python.nii.gz'
    streams = tracking_smart_debug(FA, Peaks, affine, roi_path=roi_file)
    
    if len(streams) > 0:
        print(f"--> SUCCÈS : {len(streams)} fibres.")
        save(Tractogram(streams, affine_to_rasmm=np.eye(4)), "fibres_clean.trk", header=header)
    else:
        print("--> Pas de fibres.")