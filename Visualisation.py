import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyvista as pv

#=====================
# Méthode Explicite
#=====================

# Importation des Données
#===============================================================================================================

with open("Resultat_SV_Exp.pkl", "rb") as f:
    data = pickle.load(f)

Points_Exp = data['points']      # N x 2 : (x, y)
Cells_Exp = data['cells']        # T x 3
Solutions_Exp = data['valeurs']  # t x N_points

# Mise en Forme des Données
#===============================================================================================================
# Ajouter la colonne des Z aux noeuds du maillage
Points3D_Exp = np.hstack([Points_Exp, Solutions_Exp[0][:,0].reshape(-1, 1)])  # N x 3 : (x, y, z)
# Construire un tableau plat acceptable pour PyVista pour les cellules du maillage
CellsPv_Exp = np.hstack([np.insert(tri, 0, 3) for tri in Cells_Exp]).astype(np.int64)
# Création dune liste de Cell_type (égale au nombre de triangle)
Cell_types_Exp = np.full(Cells_Exp.shape[0], pv.CellType.TRIANGLE, dtype=np.uint8)


# Affichage de la Situation Initiale
#===============================================================================================================

grid = pv.UnstructuredGrid(CellsPv_Exp, Cell_types_Exp, Points3D_Exp)

plotter = pv.Plotter(window_size=(800, 608))
scalar_bar_args = {
    "title": "Hauteur d'eau (m)"
}
plotter.add_mesh(
    grid,
    scalars=Solutions_Exp[0][:,0], # Le champ à colorer
    cmap="plasma",                 # Colormap
    show_edges=False,              # Affiche les arêtes
    edge_color="black",            # Couleur des arêtes
    line_width=0.5,                # Épaisseur des arêtes
    opacity=1.0,                   # Transparence éventuelle
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args
)
plotter.add_text("Hauteur d'eau initiale", position="upper_edge", font_size=14, color="black")
plotter.show_grid(
    color="gray",
    location='outer',              # Ou 'outer', 'all', 'front', 'back' etc.
    font_size=8                    # taille de la police liée à la grille
)
plotter.view_vector((2, -1, 0.8))  # Vue isométrique : (1, 1, 1) ou (2, 1, 1)
plotter.show()

# Animation 3D
#===============================================================================================================

### Paramètres d'animation initialisés
n_frames = len(Solutions_Exp)
dt = 0.000005

### Affichage du maillage avec scalaires initiaux
mesh = pv.UnstructuredGrid(CellsPv_Exp, Cell_types_Exp, Points3D_Exp)
mesh["champ"] = Solutions_Exp[0][:,0]  # champ au t=0

plotter = pv.Plotter(off_screen=True, window_size=(800, 608))
#plotter.open_movie("Animation_SV_Exp.mp4", framerate=10)   # Ou "animation.gif"
#plotter.open_movie("Animation_SV_Expx3.mp4", framerate=30)
plotter.add_mesh(
    mesh,
    scalars="champ",
    cmap="turbo",
    show_edges=False,
    #clim=[0.8, 1.0],
    scalar_bar_args=scalar_bar_args
)
### Affichage de la grille
plotter.show_bounds(
    xtitle="X",
    ytitle="Y", 
    ztitle="Z",
    show_xaxis=True, show_yaxis=True, show_zaxis=False,
    location="outer", grid="back", color="gray", font_size=8
)
### Vu et affigage du titre
plotter.camera_position = [(9.6, -4.8, 4), (0.0, 0.5, 0.65), (0.0, 0.0, 1.0)]
Titre_text = "Animation - Schéma Explicite (t = 0.0 s)"
Actor_text = plotter.add_text(Titre_text, position=(0.15, 0.92), viewport=True, font_size=14, color="black")
contours = mesh.contour(isosurfaces=10, scalars="champ")
actor_contours = plotter.add_mesh(contours, color="cyan", line_width=1)

for i in range(n_frames):
    # Mettre à jour le champ scalaire
    mesh["champ"] = Solutions_Exp[i][:,0]
    mesh.points[:, 2] = Solutions_Exp[i][:,0]

    # Mettre à jour le jeu d'isolignes
    plotter.remove_actor(actor_contours)
    contours = mesh.contour(isosurfaces=10, scalars="champ")
    actor_contours = plotter.add_mesh(contours, color="cyan", line_width=1)
    
    # Mettre à jour le texte
    plotter.remove_actor(Actor_text)
    current_time = i * dt
    Titre_text = f"Animation - Schéma Explicite (t = {current_time:.5f} s)"
    Actor_text = plotter.add_text(
        Titre_text,
        position=(0.15, 0.92),
        viewport=True,
        font_size=14,
        color="black"
    )
    
    plotter.render()
    # Enregistrer une frame
    plotter.write_frame()

    if i == 0:  # Frame choisie
        plotter.screenshot("apercu_video_SV_Exp.png")

plotter.close()

# Energie
#===============================================================================================================

with open("Energie_SV_Exp.pkl", "rb") as f:
    data = pickle.load(f)

Temps_Exp = data['temps'] 
Energie_Exp = data['energie']

plt.figure(figsize=(6, 6))
plt.plot(Temps_Exp, 1000*Energie_Exp, color = 'red', linestyle = '-', marker = 'o', label = 'Energie')
plt.title("Evolution de l'énergie totale - schéma explicite")
plt.xlabel('Temps (s)')
plt.ylabel('Energie (J)')
plt.xlim(Temps_Exp.min()-dt, Temps_Exp.max()+dt)
plt.legend()
plt.grid(True)
plt.savefig("Graphe_Energie_SV_Exp.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

#=====================
# Méthode Implicite
#=====================

# Importation des Données
#===============================================================================================================

with open("Resultat_SV_ImpPointFixe.pkl", "rb") as f:
    data = pickle.load(f)

Points_ImpPF = data['points']      # N x 2 : (x, y)
Cells_ImpPF = data['cells']        # T x 3
Solutions_ImpPF = data['valeurs']  # t x N_points
## Petite corection des valeurs
Solutions_ImpPF = np.roll(Solutions_ImpPF, shift=-1, axis=0)


# Mise en Forme des Données
#===============================================================================================================
# Ajouter la colonne des Z aux noeuds du maillage
Points3D_ImpPF = np.hstack([Points_ImpPF, Solutions_ImpPF[0][:,0].reshape(-1, 1)])  # N x 3 : (x, y, z)
# Construire un tableau plat acceptable pour PyVista pour les cellules du maillage
CellsPv_ImpPF = np.hstack([np.insert(tri, 0, 3) for tri in Cells_ImpPF]).astype(np.int64)
# Création dune liste de Cell_type (égale au nombre de triangle)
Cell_types_ImpPF = np.full(Cells_ImpPF.shape[0], pv.CellType.TRIANGLE, dtype=np.uint8)


# Affichage de la Situation Initiale
#===============================================================================================================

grid = pv.UnstructuredGrid(CellsPv_ImpPF, Cell_types_ImpPF, Points3D_ImpPF)

plotter = pv.Plotter()
scalar_bar_args = {
    "title": "Hauteur d'eau (m)"
}
plotter.add_mesh(
    grid,
    scalars=Solutions_ImpPF[0][:,0], # Le champ à colorer
    cmap="plasma",                 # Colormap
    show_edges=False,              # Affiche les arêtes
    edge_color="black",            # Couleur des arêtes
    line_width=0.5,                # Épaisseur des arêtes
    opacity=1.0,                   # Transparence éventuelle
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args
)
plotter.add_text("Hauteur d'eau initiale", position="upper_edge", font_size=14, color="black")
plotter.show_grid(
    color="gray",
    location='outer',              # Ou 'outer', 'all', 'front', 'back' etc.
    font_size=8                    # taille de la police liée à la grille
)
plotter.view_vector((2, -1, 0.8))  # Vue isométrique : (1, 1, 1) ou (2, 1, 1)
plotter.show()

# Animation 3D
#===============================================================================================================

### Paramètres d'animation initialisés
n_frames = len(Solutions_ImpPF)
dt = 0.000005

### Affichage du maillage avec scalaires initiaux
mesh = pv.UnstructuredGrid(CellsPv_ImpPF, Cell_types_ImpPF, Points3D_ImpPF)
mesh["champ"] = Solutions_ImpPF[0][:,0]  # champ au t=0

plotter = pv.Plotter(off_screen=True, window_size=(800, 608))
#plotter.open_movie("Animation_SV_ImpPF.mp4", framerate=10)  # Ou "animation.gif"
#plotter.open_movie("Animation_SV_ImpPFx3.mp4", framerate=30)
plotter.add_mesh(
    mesh,
    scalars="champ",
    cmap="turbo",
    show_edges=False,
    #clim=[0.8, 1.0],
    scalar_bar_args=scalar_bar_args
)
### Affichage de la grille
plotter.show_bounds(
    xtitle="X",
    ytitle="Y", 
    ztitle="Z",
    show_xaxis=True, show_yaxis=True, show_zaxis=False,
    location="outer", grid="back", color="gray", font_size=8
)
### Vu et affigage du titre
plotter.camera_position = [(9.6, -4.8, 4), (0.0, 0.5, 0.65), (0.0, 0.0, 1.0)]
Titre_text = "Animation - Schéma Implicite (Point Fixe) \n(t = 0.0 s)"
Actor_text = plotter.add_text(Titre_text, position=(0.17, 0.90), viewport=True, font_size=14, color="black")
contours = mesh.contour(isosurfaces=10, scalars="champ")
actor_contours = plotter.add_mesh(contours, color="cyan", line_width=1)

for i in range(n_frames):
    # Mettre à jour le champ scalaire
    mesh["champ"] = Solutions_ImpPF[i][:,0]
    mesh.points[:, 2] = Solutions_ImpPF[i][:,0]

    # Mettre à jour le jeu d'isolignes
    plotter.remove_actor(actor_contours)
    contours = mesh.contour(isosurfaces=10, scalars="champ")
    actor_contours = plotter.add_mesh(contours, color="cyan", line_width=1)
    
    # Mettre à jour le texte
    plotter.remove_actor(Actor_text)
    current_time = i * dt
    Titre_text = f"Animation - Schéma Implicite (Point Fixe) \n(t = {current_time:.5f} s)"
    Actor_text = plotter.add_text(
        Titre_text,
        position=(0.17, 0.90),
        viewport=True,
        font_size=14,
        color="black"
    )
    
    plotter.render()
    # Enregistrer une frame
    plotter.write_frame()

    if i == 0:  # Frame choisie
        plotter.screenshot("apercu_video_SV_ImpPF.png")

plotter.close()

# Energie
#===============================================================================================================

with open("Energie_SV_ImpPointFixe.pkl", "rb") as f:
    data = pickle.load(f)

Temps_ImpPF = data['temps'] 
Energie_ImpPF = data['energie']

plt.figure(figsize=(6, 6))
plt.plot(Temps_ImpPF, 1000*Energie_ImpPF, color = 'red', linestyle = '-', marker = 'o', label = 'Energie')
plt.title("Evolution de l'énergie totale - schéma implicite (Point fixe)")
plt.xlabel('Temps (s)')
plt.ylabel('Energie (J)')
plt.xlim(Temps_ImpPF.min()-dt, Temps_ImpPF.max()+dt)
plt.legend()
plt.grid(True)
plt.savefig("Graphe_Energie_SV_ImpPointFixe.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

#=====================
# Calcul d'erreur
#=====================

#Calcul de différence
#===============================================================================================================

Diff_Sol = []
for i in range(np.shape(Solutions_Exp)[0]):
    diff = Solutions_Exp[i] - Solutions_ImpPF[i]
    Diff_Sol.append(diff)

# Animation 3D
#===============================================================================================================

### Paramètres d'animation initialisés
n_frames = len(Diff_Sol)
dt = 0.000005

### Affichage du maillage avec scalaires initiaux
mesh = pv.UnstructuredGrid(CellsPv_ImpPF, Cell_types_ImpPF, Points3D_ImpPF)
mesh["champ"] = Diff_Sol[0][:,0]  # champ au t=0
pv.global_theme.allow_empty_mesh = True

plotter = pv.Plotter(off_screen=True, window_size=(800, 608))
#plotter.open_movie("Animation_SV_DiffExpImpPF.mp4", framerate=10)  # Ou "animation.gif"
#plotter.open_movie("Animation_SV_DiffExpImpPFx3.mp4", framerate=30)
plotter.add_mesh(
    mesh,
    scalars="champ",
    cmap="turbo",
    show_edges=False,
    clim=[np.min(Diff_Sol)-1.0e-5, np.max(Diff_Sol)+1.0e-5],
    scalar_bar_args=scalar_bar_args
)
### Affichage de la grille
plotter.show_bounds(
    xtitle="X",
    ytitle="Y", 
    ztitle="Z",
    show_xaxis=True, show_yaxis=True, show_zaxis=False,
    location="outer", grid="back", color="gray", font_size=8
)
### Vu et affigage du titre
plotter.camera_position = [(9.6, -4.8, 4), (0.0, 0.5, 0.65), (0.0, 0.0, 1.0)]
Titre_text = "Animation - Différence entre les schémas \n(t = 0.0 s)"
Actor_text = plotter.add_text(Titre_text, position=(0.17, 0.90), viewport=True, font_size=14, color="black")
contours = mesh.contour(isosurfaces=10, scalars="champ")
actor_contours = plotter.add_mesh(contours, color="darkred", line_width=1)

for i in range(n_frames):
    # Mettre à jour le champ scalaire
    mesh["champ"] = Diff_Sol[i][:,0]
    mesh.points[:, 2] = Diff_Sol[i][:,0]

    # Mettre à jour le jeu d'isolignes
    plotter.remove_actor(actor_contours)
    contours = mesh.contour(isosurfaces=10, scalars="champ")
    actor_contours = plotter.add_mesh(contours, color="darkred", line_width=1)
    
    # Mettre à jour le texte
    plotter.remove_actor(Actor_text)
    current_time = i * dt
    Titre_text = f"Animation - Différence entre les schémas \n(t = {current_time:.5f} s)"
    Actor_text = plotter.add_text(
        Titre_text,
        position=(0.17, 0.90),
        viewport=True,
        font_size=14,
        color="black"
    )
    
    plotter.render()
    # Enregistrer une frame
    plotter.write_frame()

    if i == 0:  # Frame choisie
        plotter.screenshot("apercu_video_Sv_DiffExpImpPF.png")

plotter.close()


import GeomMesh as Geo
Air_Tri = Geo.AirTri(Points_Exp, Cells_Exp)
Somme_Air = np.sum(Air_Tri)
Diff_Moyenne = []
for i in range(np.shape(Solutions_Exp)[0]):
    diff = []
    for j in range(np.shape(Solutions_Exp[0][:,0])[0]):
        diff_Moy = Diff_Sol[i][j,:]*Air_Tri[i]
        diff.append(diff_Moy)
    Somme = np.sum(diff, axis=0)
    Diff_Moyenne.append(Somme/Somme_Air)

L1_norme = []
for i in range(np.shape(Solutions_Exp)[0]):
    diff = []
    for j in range(np.shape(Solutions_Exp[0][:,0])[0]):
        abs_diff = np.abs(Diff_Sol[i][j,:])*Air_Tri[i]
        diff.append(abs_diff)
    Somme = np.sum(diff, axis=0)
    L1_norme.append(Somme/Somme_Air)

L2_norme = []
for i in range(np.shape(Solutions_Exp)[0]):
    diff = []
    for j in range(np.shape(Solutions_Exp[0][:,0])[0]):
        carré_diff = (Diff_Sol[i][j,:]**2)*Air_Tri[i]
        diff.append(carré_diff)
    Somme = np.sum(diff, axis=0)
    L2_norme.append(np.sqrt(Somme/Somme_Air))

Linf_norme = []
for i in range(np.shape(Solutions_Exp)[0]):
    max_diff = np.max(np.abs(Diff_Sol[i][j,:]), axis=0)
    Linf_norme.append(max_diff)

plt.figure(figsize=(8, 6))
plt.plot(Temps_ImpPF[:-1]  , np.array(L1_norme)[:,0], color = 'darkorange', linestyle = '-', marker = 'o', label = 'Norme L1')
plt.plot(Temps_ImpPF[:-1]  , np.array(L2_norme)[:,0], color = 'darkcyan', linestyle = '-', marker = 'o', label = 'Norme L2')
plt.plot(Temps_ImpPF[:-1]  , Linf_norme, color = 'darkmagenta', linestyle = '-', marker = 'o', label = "Norme L"+"$\infty$")
plt.title("Différence en norme entre les schémas explicite et implicite")
plt.xlabel('Temps (s)')
plt.ylabel("Hauteur d'eau (m)")
plt.xlim(Temps_ImpPF[:-1]  .min()-dt, Temps_ImpPF[:-1]  .max()+dt)
plt.legend()
plt.grid(True)
#plt.savefig("Graphe_Diff_Norme.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

Variance = []
for i in range(np.shape(Solutions_Exp)[0]):
    diff = []
    for j in range(np.shape(Solutions_Exp[0][:,0])[0]):
        carré_diff = ((Diff_Sol[i][j,:] - Diff_Moyenne[i])**2)*Air_Tri[i]
        diff.append(carré_diff)
    Somme = np.sum(diff, axis=0)
    Variance.append(Somme/Somme_Air)

EcartTyp = np.sqrt(Variance)

plt.figure(figsize=(8, 6))
plt.plot(Temps_ImpPF[:-1], np.array(Diff_Moyenne)[:,0], color = 'red', linestyle = '-', marker = 'o', label = 'Différence moyenne pondérée')
plt.fill_between(Temps_ImpPF[:-1], np.array(Diff_Moyenne)[:,0] - np.array(EcartTyp)[:,0], np.array(Diff_Moyenne)[:,0] + np.array(EcartTyp)[:,0], color="blue", alpha=0.3, label="± écart-type")
plt.title("Moyenne pondérée de la différence entre les schémas explicite et implicite")
plt.xlabel('Temps (s)')
plt.ylabel("Hauteur d'eau (m)")
plt.xlim(Temps_ImpPF[:-1].min()-dt, Temps_ImpPF[:-1].max()+dt)
plt.legend()
plt.grid(True)
#plt.savefig("Graphe_Diff_Ecart_Type.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

Max = []
for i in range(np.shape(Solutions_Exp)[0]):
    max_diff = np.max(Diff_Sol[i][j,:], axis=0)
    Max.append(max_diff)

Min = []
for i in range(np.shape(Solutions_Exp)[0]):
    min_diff = np.min(Diff_Sol[i][j,:], axis=0)
    Min.append(min_diff)

plt.figure(figsize=(8, 6))
plt.plot(Temps_ImpPF[:-1], Max, color = 'red', linestyle = '-', marker = 'o', label = 'max diff explicite/implicite')
plt.plot(Temps_ImpPF[:-1], Min, color = 'blue', linestyle = '-', marker = 'o', label = 'min diff explicite/implicite')
plt.title("Evaluation des extremums de la différence des schémas à travers le temps")
plt.xlabel('Temps (s)')
plt.ylabel("Hauteur d'eau (m)")
plt.xlim(Temps_ImpPF[:-1].min()-dt, Temps_ImpPF[:-1].max()+dt)
plt.legend()
plt.grid(True)
#plt.savefig("Graphe_Min_Max.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

Masse_tot_Exp = []
for i in range(np.shape(Solutions_Exp)[0]):
    masse = 0
    for j in range(np.shape(Solutions_Exp[0][:,0])[0]):
        masse_loc_exp = Solutions_Exp[i][j,0]*Air_Tri[i]
        masse += masse_loc_exp
    Masse_tot_Exp.append(masse)

Masse_T_Exp = np.sum(Masse_tot_Exp)/np.shape(Solutions_Exp)[0]
Delta_masse_Exp = (Masse_tot_Exp - Masse_T_Exp)/Masse_T_Exp

Masse_tot_ImpPF = []
for i in range(np.shape(Solutions_Exp)[0]):
    masse = 0
    for j in range(np.shape(Solutions_Exp[0][:,0])[0]):
        masse_loc_imppf = Solutions_ImpPF[i][j,0]*Air_Tri[i]
        masse += masse_loc_imppf
    Masse_tot_ImpPF.append(masse)

Masse_T_ImpPF = np.sum(Masse_tot_ImpPF)/np.shape(Solutions_Exp)[0]
Delta_masse_ImpPF = (Masse_tot_ImpPF - Masse_T_ImpPF)/Masse_T_ImpPF

plt.figure(figsize=(8, 6))
plt.plot(Temps_ImpPF[:-1], Delta_masse_Exp, color = 'green', linestyle = '-', marker = 'o', label = 'Explicite')
plt.plot(Temps_ImpPF[:-1], Delta_masse_ImpPF, color = 'blue', linestyle = '-', marker = 'o', label = 'Implicte')
plt.title("Comparaison relative de la masse totale à travers le temps")
plt.xlabel('Temps (s)')
plt.ylabel("Erreur relative (%)")
plt.xlim(Temps_ImpPF[:-1].min()-dt, Temps_ImpPF[:-1].max()+dt)
plt.legend()
plt.grid(True)
#plt.savefig("Graphe_Delta_Masse.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

Delta_Energie = 2*(np.abs(Energie_Exp - Energie_ImpPF))/(Energie_Exp + Energie_ImpPF)

plt.figure(figsize=(8, 6))
plt.plot(Temps_ImpPF, 100*Delta_Energie, color = 'red', linestyle = '-', marker = 'o', label = r"$\Delta$"+'Energie')
plt.title("Erreur relative d’énergie entre le schéma explicite et implicite")
plt.xlabel('Temps (s)')
plt.ylabel("Erreur relative (%)")
plt.xlim(Temps_ImpPF.min()-dt, Temps_ImpPF.max()+dt)
plt.legend()
plt.grid(True)
#plt.savefig("Graphe_Delta_Energi.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()
