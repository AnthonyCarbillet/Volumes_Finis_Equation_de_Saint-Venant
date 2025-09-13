import pygmsh
import numpy as np
from scipy import linalg as LA
from collections import defaultdict
import matplotlib.pyplot as plt

def Mesh4C(listCoo, max_cell_size = 0.25):
    # Renvoit un maillage pour une géométrie quadrangulaire, coordonnées des noeuds, liste des triangles
    #listCoo : liste des coordonnées qui forme le quadrilaitaire
    #max_cell_size : paramètre de définition de la taille caractéristique des éléments

    # Créer une instance géométrique
    with pygmsh.geo.Geometry() as geom:
        # Créer un quadrilataire
        poly = geom.add_polygon(listCoo, max_cell_size)

        # Générer le maillage
        mesh = geom.generate_mesh()

    # Extraire les points et les triangles du maillage
    points = np.array(mesh.points)
    cells = np.array(mesh.cells[-2].data)  # Les triangles sont l'indice -2

    return points, cells

#Test Mesh4C
L = [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]]

Points, Cells = Mesh4C(L)

"""
# Tracer le maillage
plt.figure(figsize=(6, 6))
plt.triplot(Points[:, 0], Points[:, 1], Cells, '-o')
plt.axis('equal')
plt.title('Maillage triangulaire avec pygmsh')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
"""

def Bord4C(Coo, tol=1e-9):
    # Renvoit la liste des indexes des coordonnées qui compose le bord d'une gémométrique quadrangulaire
    #Coo : vecteur des coordonnées du maillage d'une gémométrique quadrangulaire
    #tol : une éventuelle tolérence pour capturer les bons indexes

    # Min et Max du bord
    MinX = np.min(Coo[:,0])
    MaxX = np.max(Coo[:,0])
    MinY = np.min(Coo[:,1])
    MaxY = np.max(Coo[:,1])

    # Test des indexes
    IX1 = None
    IX2 = None
    IX3 = None
    IX4 = None
    IX1 = np.where((Coo[:,0] <= MaxX) & (Coo[:,1] == MinY))
    IX2 = np.where((Coo[:,0] == MaxX) & (Coo[:,1] <= MaxY))
    IX3 = np.where((Coo[:,0] <= MaxX) & (Coo[:,1] == MaxY))
    IX4 = np.where((Coo[:,0] == MinX) & (Coo[:,1] <= MaxY))
    
    return IX1[0], IX2[0], IX3[0], IX4[0]

#Test Bord4C
Id1, Id2, Id3, Id4 = Bord4C(Points)

X1, Y1 = Points[Id1][:,0], Points[Id1][:,1]
X2, Y2 = Points[Id2][:,0], Points[Id2][:,1]
X3, Y3 = Points[Id3][:,0], Points[Id3][:,1]
X4, Y4 = Points[Id4][:,0], Points[Id4][:,1]
#print(X)
#print(Y)

"""
# Tracer le bord
plt.figure(figsize=(6, 6))
plt.scatter(X1, Y1, color='red', label='bas')
plt.scatter(X2, Y2, color='blue', label='droit')
plt.scatter(X3, Y3, color='yellow', label='haut')
plt.scatter(X4, Y4, color='green', label='gauche')
plt.axis('equal')
plt.title('Bord du maillage')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
"""

def Voisinage(T):
    # Renvoit une liste qui contient le triangle associé à chaque arête, pour chaque triangle
    # Exemple [[ (Pour T0)[None , (arête1)], [T1 , (arête2)], [T5 , (arête3)]], ...( Pour T1 ), ...]
    #T : liste des Triangles du maillage

    # Étape 1 : construire le dictionnaire arête → triangles
    ListeAretes = defaultdict(list) #Création d'un dictionnaire vide

    for tri_idx, tri in enumerate(T):
        Aretes = [
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0]))))
                ]
        for arete in Aretes:
            ListeAretes[arete].append(tri_idx) # Liste mise à jour qui à chaque arête associe les triangles qui y sont liés
    
    # Étape 2 : déterminer les voisins pour chaque triangle
    triangle_voisins = []

    for tri_idx, tri in enumerate(T):
        Voisins = []
        Aretes = [
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0]))))
                ]
        for arete in Aretes:
            tris = ListeAretes[arete]
            voisin = [t for t in tris if t != tri_idx]
            Voisins.append((voisin[0] if voisin else None, arete))
        triangle_voisins.append(Voisins) # Liste qui contient le triangle associé à chaque arête, pour chaque triangle

    return triangle_voisins

#Test Voisinage
Tri = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
Voi = Voisinage(Tri)
#print(Voi)

def AirTri(Coo, T):
    #Renvoit la liste de l'air de triangle
    #Coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage

    tri_air = []

    for tri_idx, tri in enumerate(T):
        s1, s2, s3 = tri
        A, B, C = Coo[s1], Coo[s2], Coo[s3]
        AB = B - A
        AC = C - A
        M = np.array([AB, AC])
        Air = (1/2)*np.abs(LA.det(M))
        tri_air.append(Air)

    return np.array(tri_air)

#Test AirTri
coo = np.array([[0,0], [1,0], [1,1], [0,1], [0.5, 0.5]])
A = AirTri(coo, Tri)
#print(A)
