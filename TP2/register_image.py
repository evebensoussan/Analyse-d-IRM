#Eve BEN SOUSSAN	Cip: bene2060			IMN708
#Léhna BOUCHAMA Cip: boul5260

#--------------------------------------------------------------------------------IMPORTS------------------------------------------------------------------------------------------


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage import io
from scipy.ndimage import affine_transform, rotate
from scipy.ndimage import gaussian_filter


#--------------------------------------------------------------------------------TRANSLATE IMAGE------------------------------------------------------------------------------------------

def translate_image(I,p,q):
    """Translation d'une image I par un vecteur (p,q) avec prise en compte de l'interpolation
    p : décalage le long de l'axe des x
    q : décalage le long de l'axe des y
    mode ='nearest' : interpolation des plus proches voisins. 
        Les pixels hors de l'image sont remplis avec la valeur la plus proche"""
    return shift(I, shift=(p, q), mode='nearest')

#--------------------------------------------------------------------------------REGISTER TRANSLATION------------------------------------------------------------------------------------------

def register_translation_ssd(I,J,max_shift=10,show_progress=False):
    """
    Recalage avec la translation en minimisant SSD
    Avec :
        I : Image fixe
        J : Image mobile
        max_shift : nombre de pixels maximum pour le balayage des translations
        show_progress : affiche l'évolution en direct
    Retourne : 
        translation optimale (p,q), image recalée, SSD minimale, historique SSD
    """
    best_ssd = np.inf # SSD minimal initialisé très grand
    best_p, best_q = 0, 0 # coordonnées optimales initiales
    best_J = None # image recalée optimale
    ssd_history = [] # historique des SSD

    if show_progress:
        plt.ion() # mode interactif pour l'affichage en temps réel
        fig, ax = plt.subplots()

    # Balayage des translations possibles
    for p in range(-max_shift, max_shift + 1):
        for q in range(-max_shift, max_shift + 1):
            # Translation de J
            J_shifted = translate_image(J, p, q) # translation de l'image mobile
            # Calcul de la SSD
            current_ssd = np.sum((I - J_shifted) ** 2)  # calcul du SSD
            ssd_history.append(current_ssd)

            # Affichage si nouveau SSD minimal trouvé
            if current_ssd < best_ssd and show_progress:
                ax.clear()
                ax.imshow(I, cmap='gray')
                ax.imshow(J_shifted, cmap='hot', alpha=0.5)
                ax.set_title(f"Nouvelle meilleure SSD={current_ssd:.2f} (p={p}, q={q})")
                plt.pause(0.1)
                
            # Mise à jour du meilleur résultat
            if current_ssd < best_ssd:
                best_ssd = current_ssd
                best_p, best_q = p, q
                best_J = J_shifted.copy()

    if show_progress:
        plt.ioff()
        plt.show()
    
    return (best_p, best_q), best_J, best_ssd, ssd_history

#--------------------------------------------------------------------------------ROTATE IMAGE------------------------------------------------------------------------------------------

def rotate_image(I, theta):
    """
    Faire pivoter une image I autour du coin superieur gauche (0,0) de theta degrés
    """
    return rotate(I, angle=theta, reshape=False, order=1, mode='nearest')

#--------------------------------------------------------------------------------REGISTER ROTATION------------------------------------------------------------------------------------------

def register_rotation_ssd(I,J,max_shift=10,show_progress=False):
    """
    Recalage avec rotation et minimisation de la ssd
    Avec
        I : image fixe
        J : image mobile
        max_shift : balayage en degrés (-max_shift, max_shift)
    Retourne 
        angle optimal, image recalée, SSD minimal, historique SSD
    """

    best_ssd = np.inf
    best_theta = 0
    best_J = None
    ssd_history = []

    if show_progress:
        plt.ion()
        fig, ax = plt.subplots()

    # Balayage des translations possibles
    for theta in range(-max_shift, max_shift + 1):
        # Rotation de J
        J_rot = rotate_image(J, theta)
        # Calcul de la SSD
        current_ssd = np.sum((I - J_rot) ** 2)
        ssd_history.append(current_ssd)

        # Affichage en direct avec transparence
        #if show_progress:
        #    ax.clear()
        #    ax.imshow(I, cmap='gray')  # image fixe
        #    ax.imshow(J_rot, cmap='hot', alpha=0.5)  # image mobile semi-transparente
        #    ax.set_title(f"theta={theta}, SSD={current_ssd:.2f}")
        #    plt.pause(0.05)
        #Affichache si nouveau ssd minimal 
        if current_ssd < best_ssd and show_progress:
            ax.clear()
            ax.imshow(I, cmap='gray')
            ax.imshow(J_rot, cmap='hot', alpha=0.5)
            ax.set_title(f"Nouvelle meilleure SSD={current_ssd:.2f} (theta={theta}")
            plt.pause(0.1)

                
        # Mise à jour du meilleur résultat
        if current_ssd < best_ssd:
            best_ssd = current_ssd
            best_theta = theta
            best_J = J_rot.copy()

    if show_progress:
        plt.ioff()
        plt.show()
    
    return (best_theta), best_J, best_ssd, ssd_history

#--------------------------------------------------------------------------------REGISTER RIDIDE ------------------------------------------------------------------------------------------


def ssd(I, J):
    """Calcule la somme des différence au carrés entre 2 images."""
    return np.sum((I - J)**2)

def rotate_image_center(I, theta):
    """
    Pivote l'image autour de son centre de theta degrés en utilisant une transformation affine.
    """
    theta = np.deg2rad(theta) #conversion en radians
    h, w = I.shape
    cx, cy = w / 2, h / 2  # centre de rotation

    # Matrice de rotation 2D (dans le sens direct)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Calcul du décalage nécessaire pour tourner autour du centre
    offset = np.array([cx, cy]) - R @ np.array([cx, cy])

    # Application de la rotation
    return affine_transform(I, R, offset=offset, order=1, mode='nearest')


def register_rigid_ssd(I, J, alpha=1e-2, n_iter=200, eps=0.2, verbose=True, show_progress=True):
    """
    Recale par descente de gradient ridide (translation + rotation)
    Avec : 
        I: image fixe
        J: image mobile
        alpha: pas de gradient
        n_iter : nombre d'iterations
        eps : epsilon pour le calcul des gradients numériques 
    """
    import matplotlib.pyplot as plt

    #normalisation
    I = I.astype(np.float32) / I.max()
    J = J.astype(np.float32) / J.max()
    
    p, q, theta = 0.0, 0.0, 0.0
    ssd_history = []

    #initialisation pour suivre les modifications en direct
    if show_progress:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()  # mode interactif activé

    for i in range(n_iter):
        # Transformation de l'image rotation + translation
        J_rot = rotate_image_center(J, theta)
        J_trans = translate_image(J_rot, p, q)
        current_ssd = ssd(I, J_trans)
        ssd_history.append(current_ssd)

        # Gradients (différences finies)
        J_p = translate_image(rotate_image_center(J, theta), p + eps, q)
        grad_p = (ssd(I, J_p) - current_ssd) / eps

        J_q = translate_image(rotate_image_center(J, theta), p, q + eps)
        grad_q = (ssd(I, J_q) - current_ssd) / eps

        J_theta = rotate_image_center(J, theta + eps)
        J_theta = translate_image(J_theta, p, q)
        grad_theta = (ssd(I, J_theta) - current_ssd) / eps

        # Normalisation anti-explosion des gradients
        grad_norm = np.sqrt(grad_p**2 + grad_q**2 + grad_theta**2)
        if grad_norm > 1e2:
            grad_p /= grad_norm / 1e2
            grad_q /= grad_norm / 1e2
            grad_theta /= grad_norm / 1e2

        # Mise à jour des paramètres
        p -= alpha * grad_p
        q -= alpha * grad_q
        theta -= alpha * grad_theta

        if verbose and i % 10 == 0:
            print(f"Iter {i:03d}: SSD={current_ssd:.4f}, p={p:.3f}, q={q:.3f}, theta={theta:.3f}")

        # Affichage en temps réel
        if show_progress and i % 2 == 0:
            ax.clear()
            ax.imshow(I, cmap='gray')
            ax.imshow(J_trans, cmap='hot', alpha=0.5)  # image mobile semi-transparente
            ax.set_title(f"Iter {i} | θ={theta:.2f}°, p={p:.2f}, q={q:.2f}, SSD={current_ssd:.2f}")
            plt.pause(0.05)

        # Critère d'arrêt si convergence SSD
        if i > 5 and abs(ssd_history[-2] - ssd_history[-1]) < 1e-6:
            break

    if show_progress:
        plt.ioff()
        plt.show()

    # Résultat final
    J_reg = translate_image(rotate_image_center(J, theta), p, q)
    return (theta, p, q), J_reg, ssd_history[-1], np.array(ssd_history)


#--------------------------------------------------------------------------------RECALAGE MULTI ECHELLE ---------------------------------------------------------------------------------------

def register_rigid_multiscale(I, J, sigmas=[4, 2, 1], alpha=1e-3, n_iter=200, eps=0.2, show_progress=True, verbose=True):
    """
    Recalage multi echelle en utilisant SSD
    Avec : 
        I : image fixe
        J : image mobile
        sigmas : liste de sigma pour flouter progressivement 
    Retourne: 
        (theta, p, q), registered image, ssd_history
    """

    I = I.astype(np.float32) / I.max()
    J = J.astype(np.float32) / J.max()

    p, q, theta = 0.0, 0.0, 0.0
    ssd_history = []

    if show_progress:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))

    #boucle sur les niveaux de flou
    for level, sigma in enumerate(sigmas, 1):
        if verbose:
            print(f"\n=== Level {level}/{len(sigmas)}, sigma={sigma} ===")

        I_blur = gaussian_filter(I, sigma=sigma)#floute l'image fixe
        J_blur = gaussian_filter(J, sigma=sigma)#floute l'image mobile

        #descente de gradient sur 1 niveau
        for i in range(n_iter):
            # Transformation actuelle
            J_rot = rotate_image_center(J_blur, theta)
            J_trans = translate_image(J_rot, p, q)
            current_ssd = ssd(I_blur, J_trans)
            ssd_history.append(current_ssd)

            # Gradients numériques
            grad_p = (ssd(I_blur, translate_image(rotate_image_center(J_blur, theta), p+eps, q)) - current_ssd) / eps
            grad_q = (ssd(I_blur, translate_image(rotate_image_center(J_blur, theta), p, q+eps)) - current_ssd) / eps
            grad_theta = (ssd(I_blur, translate_image(rotate_image_center(J_blur, theta+eps), p, q)) - current_ssd) / eps

            # Normalisation anti-explosion
            grad_norm = np.sqrt(grad_p**2 + grad_q**2 + grad_theta**2)
            if grad_norm > 1e2:
                grad_p /= grad_norm / 1e2
                grad_q /= grad_norm / 1e2
                grad_theta /= grad_norm / 1e2

            # mise a jour des paramètres
            p -= alpha * grad_p
            q -= alpha * grad_q
            theta -= alpha * grad_theta

            # Affichage temps réel
            if show_progress and i % 5 == 0:
                ax.clear()
                ax.imshow(I, cmap='gray')
                ax.imshow(J_trans, cmap='hot', alpha=0.5)
                ax.set_title(f"σ={sigma} | iter={i} | θ={theta:.2f}° | p={p:.2f}, q={q:.2f} | SSD={current_ssd:.2f}")
                plt.pause(0.05)

            # Critère d’arrêt
            if i > 5 and abs(ssd_history[-2] - ssd_history[-1]) < 1e-6:
                break

        # Mise à jour du J original après chaque niveau
        J = translate_image(rotate_image_center(J, theta), p, q)

    if show_progress:
        plt.ioff()
        plt.show()

    # Image finale recalée
    J_final = translate_image(rotate_image_center(J, theta), p, q)
    return (theta, p, q), J_final, np.array(ssd_history)



#----------------------------------------------------------------------------------------MAIN---------------------------------------------------------------------------------------------------------

def main():
    # Vérifie qu'un argument a été donné
    if len(sys.argv) < 3:
        print(" Usage : python3 register_image.py image_fixe.jpg image_mobile.jpg")
        sys.exit(1)

    # Récupère le nom du fichier
    image_name = sys.argv[1]
    if not image_name.lower().endswith('.jpg'):
        image_name += '.jpg'

    #charge image fixe
    print(f" Chargement de {image_name} ...")
    I = io.imread(image_name, as_gray=True)


#============================================Translation==================================================



    # On crée une image J artificiellement translatée
    J_list = [translate_image(I, 3, -2),
              translate_image(I, -1, 4),
              translate_image(I, 2, 2)]

    # On fait le recalage
    for idx, J in enumerate(J_list, start=1):
        print(f"\n Test avec J{idx}")
        (p_opt, q_opt), J_reg, ssd_min, ssd_history = register_translation_ssd(I, J, max_shift=5, show_progress=True)
        print(f" Translation optimale pour J{idx} : p={p_opt}, q={q_opt}, SSD={ssd_min:.2f}")

        print(f"Translation optimale : (p={p_opt}, q={q_opt}), SSD = {ssd_min:.2f}")

        # Affiche résultat final
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(I, cmap='gray'); axes[0].set_title('Image fixe (I)')
        axes[1].imshow(J, cmap='gray'); axes[1].set_title('Image mobile (J)')
        axes[2].imshow(J_reg, cmap='gray'); axes[2].set_title('Image recalée (J_reg)')
        plt.show()

        #plot energie
        plt.figure(figsize=(6,4))
        plt.plot(ssd_history, '-o', color='royalblue')
        plt.title("Évolution du SSD pendant le recalage")
        plt.xlabel("Itération")
        plt.ylabel("SSD")
        plt.grid(True)
        plt.show()

    

#============================================Rotation==================================================



    # On crée une image J artificiellement rotée
    J_list = [rotate_image(I, 20),
              rotate_image(I, 70),
              rotate_image(I, 120)]

    # On fait le recalage
    for idx, J in enumerate(J_list, start=1):
        print(f"\n Test avec J{idx}")
        (theta_opt), J_reg, ssd_min, ssd_history = register_rotation_ssd(I, J, max_shift=180, show_progress=True)
        print(f" Rotation optimale pour J{idx} : theta ={theta_opt}, SSD={ssd_min:.2f}")

        # Affiche résultat final
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(I, cmap='gray'); axes[0].set_title('Image fixe (I)')
        axes[1].imshow(J, cmap='gray'); axes[1].set_title('Image mobile (J)')
        axes[2].imshow(J_reg, cmap='gray'); axes[2].set_title('Image recalée (J_reg)')
        plt.show()

        #plot energie
        plt.figure(figsize=(6,4))
        plt.plot(ssd_history, '-o', color='royalblue')
        plt.title("Évolution du SSD pendant le recalage")
        plt.xlabel("Itération")
        plt.ylabel("SSD")
        plt.grid(True)
        plt.show()



#============================================Rotation + Translation==================================================



    images_mobiles = sys.argv[2:]
    for idx,image_mobile in enumerate(images_mobiles):
        J = io.imread(image_mobile, as_gray=True)
        print(f"\n Test avec BrainMRI_{idx+1}")
        results = []
        
        (theta_opt, p_opt, q_opt), J_reg, ssd_min, ssd_history = register_rigid_ssd(
        I, J, alpha=0.1, n_iter=200, eps=0.002, verbose=True)

        print(f"Résultat optimal : theta={theta_opt:.2f}, p={p_opt:.2f}, q={q_opt:.2f}, SSD={ssd_min:.2f}")

        # Affichage final
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(I, cmap='gray'); axes[0].set_title("Image fixe")
        axes[1].imshow(J, cmap='gray'); axes[1].set_title("Image mobile")
        axes[2].imshow(J_reg, cmap='hot', alpha=0.5); axes[2].set_title("Image recalée")
        plt.show()

        initial_ssd = ssd(I, J)
        improvement = 100 * (1 - ssd_min / initial_ssd)
        results.append((image_mobile, theta_opt, p_opt, q_opt, ssd_min, improvement))

        # Courbe de convergence
        plt.figure()
        plt.plot(ssd_history, '-o')
        plt.title(f"Convergence SSD — {image_mobile}")
        plt.xlabel("Itération")
        plt.ylabel("SSD")
        plt.show()



#============================================Recalage rigide améliorée==================================================



# On crée 3 cas de transformations rigides
    tests = [
        {"theta": 10, "p": 5, "q": -3},
        {"theta": -15, "p": -10, "q": 8},
        {"theta": 25, "p": 12, "q": 6}
    ]

    # On fait le recalage
    for i, t in enumerate(tests):
        print(f"\n==== Test {i+1}: theta={t['theta']}, p={t['p']}, q={t['q']} ====")
        J_test = translate_image(rotate_image_center(I, t["theta"]), t["p"], t["q"])

        (theta_opt, p_opt, q_opt), J_final, ssd_total = register_rigid_multiscale(
            I, J_test, sigmas=[4, 2, 1], alpha=0.05, n_iter=100, eps=0.2, show_progress=True
        )

        print(f"Résultat final : theta={theta_opt:.2f}, p={p_opt:.2f}, q={q_opt:.2f}")

        # Affichage final
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(I, cmap='gray'); axes[0].set_title('Image fixe (I)')
        axes[1].imshow(J_test, cmap='gray'); axes[1].set_title('Image mobile (J)')
        axes[2].imshow(J_final, cmap='gray'); axes[2].set_title('Image recalée (J_reg)')
        plt.show()

        # Courbe SSD
        plt.figure(figsize=(6, 4))
        plt.plot(ssd_total, '-o', color='royalblue')
        plt.title("Évolution du SSD pendant le recalage multi-échelle")
        plt.xlabel("Itération")
        plt.ylabel("SSD")
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    main()