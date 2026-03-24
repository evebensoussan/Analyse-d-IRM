#Eve BEN SOUSSAN	Cip: bene2060			IMN708
#Léhna BOUCHAMA Cip: boul5260

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pour affichage 3D

# Génération d'une grille 3D régulière de points
def generate_grid(n=20):
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(1))
    points = np.vstack((x.flatten(), y.flatten(), z.flatten(), np.ones(x.size)))
    return points

# Fonction pour afficher des points 3D

def plot_points(points, color='k', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :], c=color, marker='o')
    return ax

# Création et affichage de la grille initiale (Figure 1a)

if __name__ == "__main__":
    points = generate_grid()
    ax = plot_points(points)
    ax.set_title("Figure 1a — Grille 3D régulière")
    plt.show()



# Transformation rigide (rotation + translation) 

def trans_rigide(theta, omega, phi, p, q, r):
    # Rotation autour de X
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    # Rotation autour de Y
    Ry = np.array([[np.cos(omega), 0, np.sin(omega), 0],
                   [0, 1, 0, 0],
                   [-np.sin(omega), 0, np.cos(omega), 0],
                   [0, 0, 0, 1]])

    # Rotation autour de Z
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                   [np.sin(phi), np.cos(phi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Translation
    T = np.array([[1, 0, 0, p],
                  [0, 1, 0, q],
                  [0, 0, 1, r],
                  [0, 0, 0, 1]])

    # Transformation rigide complète : T * Rz * Ry * Rx
    M = T @ Rz @ Ry @ Rx
    return M



### test sur grille de point pour obtenir la figure 1.b

theta, omega, phi = np.pi/4, np.pi/6, np.pi/3 # angles rotation
p, q, r = 5, 5, 10                   # translation

M = trans_rigide(theta, omega, phi, p, q, r) 
transformed_points = M @ points

ax = plot_points(points, 'k')
plot_points(transformed_points, 'r', ax)
ax.set_title("Figure 1b — Transformation rigide")
plt.show()




# Transformation de similitude (rigide + homothétie)

def similitude(theta, omega, phi, p, q, r, s):
    M = trans_rigide(theta, omega, phi, p, q, r)
    S = np.diag([s, s, s, 1])    # matrice d'échelle homogène
    return M @ S


### test sur figure 1.b pour obtenir figure 1.c

s = 0.2
M = similitude(0, 0, 0, 0, 0, 0, s)
transformed_points = M @ points

ax = plot_points(points, 'k')
plot_points(transformed_points, 'r', ax)
ax.set_title("Figure 1c — Transformation de similitude")
plt.show()

# Analyse de matrices 3D données (rigide / similitude / affine)

def analyze_matrix(M, name="M"):
    """Analyse la nature d'une matrice de transformation 3D."""
    A = M[:3, :3]       # sous-matrice rotation/échelle
    RtR = A.T @ A
    identity = np.eye(3)
    ortho_err = np.linalg.norm(RtR - identity)
    det = np.linalg.det(A)
    U, S, Vt = np.linalg.svd(A)
    svals = S

    print(f"\n=== Analyse de {name} ===")
    print(M)
    print(f"\n- Erreur d'orthogonalité : {ortho_err:.6f}")
    print(f"- Déterminant : {det:.6f}")
    print(f"- Valeurs singulières : {np.round(svals, 4)}")

    classification = "inconnue"
    if ortho_err < 1e-6 and abs(det - 1) < 1e-6:
        classification = "Transformation rigide (rotation + translation)"
    elif np.allclose(svals[0], svals[1], atol=1e-3) and np.allclose(svals[1], svals[2], atol=1e-3):
        classification = "Transformation de similitude (rotation + homothétie)"
    else:
        classification = "Transformation affine (déformation non uniforme ou cisaillement)"
    print(f"→ Type probable : {classification}")
    return classification


def apply_transform(M, points):
    transformed = M @ points
    if not np.allclose(transformed[3, :], 1):
        transformed = transformed / transformed[3, :]
    return transformed


# Matrices données
M1 = np.array([
    [0.9045, -0.3847, -0.1840, 10.0000],
    [0.2939,  0.8750, -0.3847, 10.0000],
    [0.3090,  0.2939,  0.9045, 10.0000],
    [0.0,     0.0,     0.0,    1.0000]
])

M2 = np.array([
    [-0.0000, -0.2598,  0.1500, -3.0000],
    [ 0.0000, -0.1500, -0.2598,  1.5000],
    [ 0.3000,  0.0000,  0.0000,  0.0000],
    [ 0.0,     0.0,     0.0,     1.0000]
])

M3 = np.array([
    [ 0.7182, -1.3727, -0.5660,  1.8115],
    [-1.9236, -4.6556, -2.5512,  0.2873],
    [-0.6426, -1.7985, -1.6285,  0.7404],
    [ 0.0,      0.0,     0.0,     1.0000]
])


# Visualisation et analyse des trois matrices
fig = plt.figure(figsize=(15, 5))
matrices = [M1, M2, M3]
titles = ["M1", "M2", "M3"]

for i, M in enumerate(matrices):
    transformed_points = apply_transform(M, points)
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    plot_points(points, 'k', ax)
    plot_points(transformed_points, 'r', ax)
    ax.set_title(f"Transformation {titles[i]}")
    analyze_matrix(M, titles[i])

plt.tight_layout()
plt.show()
