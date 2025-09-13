import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import pickle
import GeomMesh as Goe
import GetFVMDataSV as FVM
import matplotlib.tri as tri
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# Création de la géométrie et de son maillage
#===============================================================================================================
C = [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]]
Points, Cells = Goe.Mesh4C(C,0.1)

#Points = np.array([[0,0,0],[1,0,0],[0,1,0]]) #Triangle Unité
#Cells = np.array([[0,1,2]])

"""
plt.figure(figsize=(6, 6))
plt.triplot(Points[:, 0], Points[:, 1], Cells, '-o')
plt.axis('equal')
plt.title('Maillage triangulaire avec pygmsh')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
"""

Points = np.delete(Points, -1, axis=1) #on supprime les coordonnés en Z

# Condition initiale d'une goute d'eau
#===============================================================================================================
Z_f = FVM.Fond(Points)
Zf = FVM.VolumeMoy(Cells, Z_f)
Init = FVM.InitGoute(Points, 0.8, Z_f, 1.2, -1, -1, 0.4)
u_0, v_0 = np.zeros(len(Z_f)), np.zeros(len(Z_f))
U0 = FVM.VolumeInit(Cells, Init, u_0, v_0)
#print(U0)

# Résolution des volumes
#===============================================================================================================
Nn = 4 #652
pas = 0.000005 #0.000005
#SolutionFV = FVM.ResolutionExp(Points, Cells, U0, pas, Nn*pas, Zf) # schéma explicite
#SolutionFV = FVM.ResolutionImp(Points, Cells, U0, pas, Nn*pas, Zf, 10) # schéma implicite
SolutionFV = FVM.ResolutionImpGlo(Points, Cells, U0, pas, Nn*pas, Zf, 30) # schéma implicite global

Solution = []
N_Noeud = len(Points)
for i in range(len(SolutionFV[0])):
    S = FVM.MailleInterpo(SolutionFV[0][i], Cells, N_Noeud)
    Solution.append(S)

# Sauvegarde des Données
#===============================================================================================================
"""
data = {
    'points': Points,         # coordonnées des nœuds
    'cells': Cells,           # connectivité du maillage
    'valeurs': Solution  # tableau T x N_points
}

with open("Resultat_SV_ImpPointFixe.pkl", "wb") as f:
    pickle.dump(data, f)

"""
# Affichage de la Situation Initiale
#==============================================================================================================

H = Solution[0][:,0]

triang = tri.Triangulation(Points[:, 0], Points[:, 1])
# Visualisation 3D de la hauteur d'eau
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
cmap = plt.cm.plasma

# Tracé de la surface 3D
surf = ax.plot_trisurf(triang, H, cmap=cmap, edgecolor='none')
# Ajout d'une barre de couleur
cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
m = plt.cm.ScalarMappable(cmap=cmap)
m.set_array(H)
fig.colorbar(m, cax=cbar_ax, label='Hauteur d\'eau (m)')
# Configuration de l'affichage
ax.set_title('Hauteur d\'eau Initiale')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Hauteur d\'eau (m)')
ax.set_zlim(0.0, 2.5)
ax.view_init(elev=25, azim=-45)  # Ajustement de la vue
#plt.savefig("Condition_Initiale_SV.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

# Animatimation 3D
#==============================================================================================================
# Initialiser la figure et l'axe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Fonction d'initialisation
def init():
    triang = ax.plot_trisurf(Points[:,0], Points[:,1], Solution[0][:,0], triangles=Cells, cmap='plasma', edgecolor='none')
    ax.set_title('Hauteur d\'eau en 3D')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Hauteur d\'eau (m)')
    ax.set_zlim(0.0, 2.5) # Ajuster les limites de l'axe Z selon vos données
    ax.view_init(elev=25, azim=-45)  # Ajustement de la vue
    return fig,
# Fonction d'animation
def animate(i):
    ax.clear()
    triang = ax.plot_trisurf(Points[:, 0], Points[:, 1], Solution[i][:,0], triangles=Cells, cmap='plasma', edgecolor='none')
    ax.set_title('Hauteur d\'eau en 3D')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Hauteur d\'eau (m)')
    ax.set_zlim(0.0, 2.5)  # Ajuster les limites de l'axe Z selon vos données
    ax.view_init(elev=25, azim=-45)  # Ajustement de la vue
    return fig,
# Créer l'animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=Nn+1, interval=100, blit=False)
cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
m = plt.cm.ScalarMappable(cmap=cmap)
m.set_array(H)
fig.colorbar(m, cax=cbar_ax, label='Hauteur d\'eau (m)')

# Afficher l'animation
plt.show()

# Analyse d'Energie
#==============================================================================================================
Energie = SolutionFV[1]
Time = np.arange(0, (Nn+2)*pas, pas)

plt.figure(figsize=(6, 6))
plt.plot(Time, Energie, color = 'red', linestyle = '-', marker = 'o', label = 'Energie')
plt.title("Evolution de l'énergie totale")
plt.xlabel('Temps (s)')
plt.ylabel('Energie (J)')
plt.xlim(Time.min()-pas, Time.max()+pas)
plt.legend()
plt.grid(True)
plt.show()

# Sauvegarde des Données
#===============================================================================================================
"""
data = {
    'temps': Time,         # Echelle temporelle
    'energie': Energie     # Valeurs d'énergie
}

with open("Energie_SV_ImpPointFixe.pkl", "wb") as f:
    pickle.dump(data, f)
"""
