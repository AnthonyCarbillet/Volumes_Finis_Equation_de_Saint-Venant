import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import GeomMesh as Geo

def Fond(coo):
    #Renvoit les valeurs du fond sur les noeuds du maillage
    #coo : vecteur contenant les coordonnées des noeuds du maillage
    
    x, y = coo[:,0], coo[:,1]
    #Z = 0.05*x + 0.05*y
    Z = 0*x + 0*y

    return Z

def VolumeMoy(T, X):
    # Renvoit les valeurs moyennes d'une variables pour chaque triangle
    #T : liste des Triangles du maillage
    #X : variable courrante évaluée sur le maillage

    paramètres = {}
    C = len(T)
    for i in range(0, C):
        s = T[i]
        paramètres['X' + str(i)] = (X[s[0]] + X[s[1]] + X[s[2]])/3

    varX = np.array([valeur for clé, valeur in paramètres.items() if clé.startswith("X")])

    return varX

def InitGoute(coo, eta, z, A, xg, yg, sigma):
    #Renvoit les valeurs de la condition initiale pour une goutte d'eau
    #coo : vecteur contenant les coordonnées des noeuds du maillage
    #eta : valeur de la hauteur d'eau initiale par fond plat
    #z : les valeurs du fond sur les noeuds du maillage
    #A : ampplitude de la goutte d'eau
    #xg, yg : les coordonnées de la goutte d'eau
    #sigma : largeur de la goutte d'eau

    x, y = coo[:,0], coo[:,1]
    Eta_0 = eta*np.ones(len(z))
    Eta = Eta_0 + A*np.exp(-((x-xg)*(x-xg) + (y-yg)*(y-yg))/(sigma*sigma))
    Goutte = Eta - z

    return Goutte

def VolumeInit(T, h, u, v):
    # Renvoit les valeurs moyennes des variables d'état pour chaque triangle
    #T : liste des Triangles du maillage
    #h : hauteur d'eau initiale pour chaque noeud
    #u : vitesse de l'eau en x initiale pour chaque noeud
    #v : vitesse de l'eau en y initiale pour chaque noeud

    paramètres = {}
    C = len(T)
    for i in range(0, C):
        s = T[i]
        paramètres['h' + str(i)] = (h[s[0]] + h[s[1]] + h[s[2]])/3
        paramètres['u' + str(i)] = (u[s[0]] + u[s[1]] + u[s[2]])/3
        paramètres['v' + str(i)] = (v[s[0]] + v[s[1]] + v[s[2]])/3

    varH = np.array([valeur for clé, valeur in paramètres.items() if clé.startswith("h")])
    varU = np.array([valeur for clé, valeur in paramètres.items() if clé.startswith("u")])
    varV = np.array([valeur for clé, valeur in paramètres.items() if clé.startswith("v")])

    return varH, varU, varV

"""
#Test VolumeInit
Coo = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
Tri = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
H = np.array([1, 1, 3, 3, 2])
U = np.array([1, 1, 3, 3, 2])
V = np.array([1, 1, 3, 3, 2])

#Test VolumeInit
MoyTri = VolumeInit(Tri, H, U, V)
#print(MoyTri)
"""

def NormalVec(coo, T):
    # Renvoit une liste des vecteurs normaux de chaque arête, pour chaque triangle
    #coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage

    tri_normal = []
    for tri in T:
        s1, s2, s3 = tri

        A = np.array(coo[s1])
        B = np.array(coo[s2])
        C = np.array(coo[s3])
        # Barycentre
        center = (A + B + C) / 3

        # Arêtes du triangle
        aretes = [(A, B), (B, C), (C, A)]
        normal = []

        for P1, P2 in aretes:
            t = P2 - P1  # vecteur tangentiel (arête)
            n = np.array([ -t[1], t[0] ])  # rotation 90° anti-horaire
            n_unit = n / np.linalg.norm(n)

            # Correction d'orientation si la normale ne pointe pas vers l'extérieur
            midpoint = 0.5 * (P1 + P2)
            to_outside = midpoint + 0.01 * n_unit - center
            if np.dot(to_outside, n_unit) < 0:
                n_unit = -n_unit

            normal.append(n_unit)

        tri_normal.append(normal)
    
    return tri_normal

"""
#Test NormalVec
Norma = NormalVec(Coo, Tri)
#print(Norma)
"""

def LongArete(A, B, C):
    # Renvoit la longeur des arêtes d'un triangle ABC
    #A, B, C : les coorodonnées des sommets du triangle ABC
    
    Longeur = []
    aretes = [(A, B), (B, C), (C, A)]
    for P1, P2 in aretes:
        arete = P2 - P1  # vecteur tangentiel (arête)
        l = np.sqrt( arete[0]*arete[0] + arete[1]*arete[1])
        Longeur.append(l)
    
    return Longeur

"""
#Test LongArete
a = Coo[0]
b = Coo[1]
c = Coo[4]
Long = LongArete(a, b, c)
#print(Long)
"""
def GloEnergie(U, tri_air, g = 9.81):
    #Renvoit l'énergie pour tout les triangles donnés via ses variables d'états
    #U : vecteur des variables d'états à l'instant présent pour un triangle donné
    #tri_air : vecteur des airs des triangles donnés
    #g : accélération de la pesanteur

    h , hu, hv = U[:, 0].copy(), U[:, 1], U[:, 2]
    # Mise à zéro des petites valeurs de h

    u = np.where(h < 1e-5, hu / h, 0.0)
    v = np.where(h < 1e-5, hv / h, 0.0)

    Energ = (tri_air/2)*(h*(u*u + v*v) + g*h*h)

    return Energ

# ===========================================================================================================================================
# Résolution Explicite ======================================================================================================================
# ===========================================================================================================================================
# ===========================================================================================================================================

def Flux(U, n, g = 9.81):
    # Renvoit le flux sortant d'un triangle donné sur l'arête normale à n
    #U : vecteur des variables d'état pour un triangle donné
    #n : vecteur normal à une des arêtes du triangle
    #g : accélération de la pesanteur
    
    if U[0] > 1e-5:
        h, u, v = U[0], U[1]/U[0], U[2]/U[0]
    else:
        h, u, v = U[0], 0, 0

    f1 = np.array([h*u, h*u*u + 0.5*g*h*h, h*u*v])
    f2 = np.array([h*v, h*u*v, h*v*v + 0.5*g*h*h])
    flux = n[0]*f1 + n[1]*f2
    
    return flux

"""
#Test Flux
H_u, H_v = MoyTri[0]*MoyTri[1], MoyTri[0]*MoyTri[2]
UtestI = np.array([MoyTri[0][0], H_u[0], H_v[0]])
n0 = Norma[0][1]
Ff = Flux(UtestI, n0)
#print(Ff)
#print(n0)
"""

def LambdaMax(Ui, Uj, n, g = 9.81):
    # Renvoit le coefficient Lambda_max de la formule de flux de Rusanov
    #Ui : vecteur des variables d'état pour un triangle i
    #Uj : vecteur des variables d'état pour un triangle j
    #n : vecteur normal à une des arêtes du triangle
    #g : accélération de la pesanteur

    Veci = np.array([Ui[1]/Ui[0], Ui[2]/Ui[0]])
    Vecj = np.array([Uj[1]/Uj[0], Uj[2]/Uj[0]])
    VecI = np.abs(np.dot(Veci, n))
    VecJ = np.abs(np.dot(Vecj, n))

    L_max = np.max([np.abs(VecI) + np.sqrt(g*max(Ui[0], 1e-8)), np.abs(VecJ) + np.sqrt(g*max(Uj[0], 1e-8))])

    return L_max

"""
#Test LambdaMax
UtestJ = np.array([MoyTri[1][0], H_u[1], H_v[1]])
Lambda_max = LambdaMax(UtestI, UtestJ, n0)
#print(Lambda_max)
"""

def EtatMirroir(U, n):
    #Renvoit un vecteur d'état mirroir pour une condition aux limites de mur rigide
    #U : vecteur d'état d'un volume de control sur un bord
    #n : vecteur normal

    h, hu, hv = U
    if h > 1-5:
        u_vec = np.array([hu, hv]) / h
    else:
        u_vec = np.array([0, 0])
    un = np.dot(u_vec, n)
    u_mirroir = u_vec - 2 * un * n
    hu_m, hv_m = h * u_mirroir

    return np.array([h, hu_m, hv_m])

def FluxRusanovExp(coo, T, h, hu, hv, Voisins, Normals, Z_fond):
    # Renvoit une liste qui contient la valeur du flux pour chaque triangle pour un schéma explicite
    #coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage
    #h : hauteur d'eau initiale pour chaque noeud
    #hu : débit de l'eau en x initiale pour chaque noeud
    #hv : débit de l'eau en y initiale pour chaque noeud
    #Voisins : liste qui contient le triangle associé à chaque arête, pour chaque triangle
    #Normals : liste des vecteurs normaux de chaque arête, pour chaque triangle
    #Z_fond : Liste des valeurs du fond pour chaque triangle

    tri_Flux = []
    for tri_idx, tri in enumerate(T):
        s1, s2, s3 = tri
        A = np.array(coo[s1])
        B = np.array(coo[s2])
        C = np.array(coo[s3])

        Long = LongArete(A, B, C)
        Ui = np.array([h[tri_idx], hu[tri_idx], hv[tri_idx]])
        Flux_Rusa = np.zeros(3)
        for i in range(3):
            n = Normals[tri_idx][i]
            Voi = Voisins[tri_idx][i][0]
            zi = Z_fond[tri_idx]
            if Voi != None :
                Uj = np.array([h[Voi], hu[Voi], hv[Voi]])
                zj = Z_fond[Voi]
            else:
                Uj = EtatMirroir(Ui, n)
                zj = zi

            L_max = LambdaMax(Ui, Uj, n)
            Rusanov = 0.5*(Flux(Ui,n) + Flux(Uj, n)) - 0.5*L_max*(Uj - Ui)
            Flux_Rusa += Long[i]*Rusanov
        
        tri_Flux.append(Flux_Rusa)
    
    return tri_Flux

"""
#Test FluxRusanovExp
Voi = Geo.Voisinage(Tri)
Flux_Rusa = FluxRusanovExp(Coo, Tri, MoyTri[0], H_u, H_v, Voi, Norma)
#print(Flux_Rusa)
"""

def TriSource(T, h, hu, hv, z = np.array([0.0, 0.0]), g = 9.81, rho = 1000, n_Maning = 0.02):
    #Revnoit une liste qui contient la valeur des termes sources pour chaque triangle
    #T : liste des Triangles du maillage
    #h : hauteur d'eau initiale pour chaque noeud
    #hu : débit de l'eau en x initiale pour chaque noeud
    #hv : débit de l'eau en y initiale pour chaque noeud
    #z : vecteur des dérivés du fond (supposé linéaire ici)
    #g : accélération de la pesanteur
    #rho : densité de l’eau
    #n_Maning : coefficient de Manning

    tri_source = []
    for i in range(len(T)):
        H = h[i]
        Hu = hu[i]
        Hv = hv[i]
        if H > 1e-5:
            u, v = Hu/H, Hv/H
        else:
            u, v = 0, 0
        Tau = np.array([rho*g*(n_Maning*n_Maning)*u*((np.sqrt(u*u + v*v))/(np.cbrt(H**4))), rho*g*(n_Maning*n_Maning)*v*((np.sqrt(u*u + v*v))/(np.cbrt(H**4)))])
        Source = np.array([ 0*H, -g*H*z[0] -Tau[0], -g*H*z[1] -Tau[1]])
        tri_source.append(Source)
    
    return tri_source

"""
#Test TriSource
Sources = TriSource(Tri, MoyTri[0], H_u, H_v)
#print(Sources)
"""

def TriReso(U, dt, tri_air, tri_flux, tri_source):
    #Renvoit la mise à jour des variables d'états pour un triangle donné
    #U : vecteur des variables d'états à l'instant présent pour un triangle donné
    #dt : pas de temps
    #tri_air : air du triangle donné
    #tri_flux : flux du triangle donné
    #tri_source : termes sources du triangle donné

    Udt = U - (dt/tri_air)*tri_flux + dt*tri_source

    return Udt

def TriEnergie(U, tri_air, g = 9.81):
    #Renvoit l'énergie pour un triangle donné via ses variables d'états
    #U : vecteur des variables d'états à l'instant présent pour un triangle donné
    #tri_air : air du triangle donné
    #g : accélération de la pesanteur

    if U[0] > 1e-5:
        h, u, v = U[0], U[1]/U[0], U[2]/U[0]
    else:
        h, u, v = U[0], 0, 0
    
    Energ = (tri_air/2)*(h*(u*u + v*v) + g*h*h)

    return Energ

def ResolutionExp(Coo, T, Uinit, dt, Time, Z_fond):
    #Renvoit la résolution explicite des variables d'états et l'energie pour tout instant dt entre 0 et le temps final Time
    #Coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage
    #Uinit : vecteur des valeurs moyennes des variables d'états initiales pour chaque triangle
    #dt : pas de temps
    #Time : limite de temps finale
    #Z_fond : Liste des valeurs du fond pour chaque triangle

    H, Hu, Hv = Uinit[0], Uinit[0]*Uinit[1], Uinit[0]*Uinit[2]
    U_now = np.vstack((H, Hu, Hv)).T
    Tri_air = Geo.AirTri(Coo, T)
    Voisins = Geo.Voisinage(T)
    Normals = NormalVec(Coo, T)

    Historia = []
    Historia.append(U_now)
    EnergieTot = []
    E0 = GloEnergie(U_now, Tri_air)
    EnergieTot.append(np.sum(E0))
    for t in tqdm(np.arange(0, Time+dt, dt)):
        Tri_flux = FluxRusanovExp(Coo, T, U_now[:,0], U_now[:,1], U_now[:,2], Voisins, Normals, Z_fond)
        Tri_source = TriSource(T, U_now[:,0], U_now[:,1], U_now[:,2])
        U_futur = []
        Energie_t = []
        for tri_idx, tri in enumerate(T):
            u_now = U_now[tri_idx]

            tri_air = Tri_air[tri_idx]
            tri_flux = Tri_flux[tri_idx]
            tri_source = Tri_source[tri_idx]
            u_futur = TriReso(u_now, t, tri_air, tri_flux, tri_source)
            U_futur.append(u_futur)
            tri_energie = TriEnergie(u_futur, tri_air)
            Energie_t.append(tri_energie)

        Energie_t = np.array(Energie_t)
        EnergieTot.append(np.sum(Energie_t))
        U_futur = np.array(U_futur)
        Historia.append(U_futur)
        U_now = U_futur.copy()

    EnergieTot = np.array(EnergieTot)

    return Historia, EnergieTot

# ===========================================================================================================================================
# Résolution Implicite Locale ===============================================================================================================
# ===========================================================================================================================================
# ===========================================================================================================================================

def FluxRusanovImp(coo, tri, h, hu, hv, U_guess, Voisins, Normals, Z_fond):
    # Renvoit une liste qui contient la valeur du flux pour chaque triangle pour un schéma implicite
    #coo : vecteur contenant les coordonnées des noeuds du maillage
    #tri : triangle du maillage
    #h : hauteur d'eau pour chaque noeud
    #hu : débit de l'eau en x pour chaque noeud
    #hv : débit de l'eau en y pour chaque noeud
    #U_guess : stade itératif du vecteur des variables d'états 
    #Voisins : liste qui contient le triangle associé à chaque arête
    #Normals : liste des vecteurs normaux de chaque arête
    #Z_fond : Liste des valeurs du fond

    s1, s2, s3 = tri
    A = np.array(coo[s1])
    B = np.array(coo[s2])
    C = np.array(coo[s3])

    Long = LongArete(A, B, C)
    Ui = np.array([h, hu, hv])
    Flux_Rusa = np.zeros(3)
    for i in range(3):
        n = Normals[i]
        Voi = Voisins[i][0]
        zi = Z_fond
        if Voi != None :
            Uj = np.array([U_guess[Voi,:]])

        else:
            Uj = EtatMirroir(Ui, n)

        L_max = LambdaMax(Ui, Uj.flatten(), n)
        Rusanov = 0.5*(Flux(Ui,n) + Flux(Uj.flatten(), n)) - 0.5*L_max*(Uj.flatten() - Ui)
        Flux_Rusa += Long[i]*Rusanov

    return Flux_Rusa

def ResolutionImp(Coo, T, Uinit, dt, Time, Z_fond, iter_max):
    #Renvoit la résolution implicite des variables d'états et l'energie pour tout instant dt entre 0 et le temps final Time
    #Coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage
    #Uinit : vecteur des valeurs moyennes des variables d'états initiales pour chaque triangle
    #dt : pas de temps
    #Time : limite de temps finale
    #Z_fond : Liste des valeurs du fond pour chaque triangle
    #iter_max : nombe d'itération maximal pour le point fixe de la méthode implicite

    H, Hu, Hv = Uinit[0], Uinit[0]*Uinit[1], Uinit[0]*Uinit[2]
    U_now = np.vstack((H, Hu, Hv)).T
    Tri_air = Geo.AirTri(Coo, T)
    Voisins = Geo.Voisinage(T)
    Normals = NormalVec(Coo, T)

    Historia = []
    Historia.append(U_now)
    EnergieTot = []
    E0 = GloEnergie(U_now, Tri_air)
    EnergieTot.append(np.sum(E0))
    for t in tqdm(np.arange(0, Time+dt, dt)):
        U_guess = U_now.copy()
        for k in range(iter_max):            
            U_futur = np.zeros_like(U_now)
            Energie_t = []
            for tri_idx, tri in enumerate(T):
                U_tri = U_guess[tri_idx,:]
                Tri_flux = FluxRusanovImp(Coo, tri, U_tri[0], U_tri[1], U_tri[2], U_guess, Voisins[tri_idx], Normals[tri_idx], Z_fond[tri_idx])
                Tri_Source = TriSource(np.array([0]), U_tri[0].reshape(1, -1), U_tri[1].reshape(1, -1), U_tri[2].reshape(1, -1)) 
                U_futur[tri_idx] = TriReso(U_now[tri_idx], t, Tri_air[tri_idx], Tri_flux, Tri_Source[0].flatten())
                tri_energie = TriEnergie(U_futur[tri_idx], Tri_air[tri_idx])
                Energie_t.append(tri_energie)

            if np.linalg.norm(U_futur - U_guess, ord=np.inf) < 1e-5:
                break
            U_guess[:] = U_futur

        U_now[:] = U_guess.copy()
        Historia.append(U_guess)
        Energie_t = np.array(Energie_t)
        EnergieTot.append(np.sum(Energie_t))

    EnergieTot = np.array(EnergieTot)

    return Historia, EnergieTot

# ===========================================================================================================================================
# Résolution Implicite Globale ==============================================================================================================
# ===========================================================================================================================================
# ===========================================================================================================================================

def Jaco_S(Ui, z = np.array([0.0, 0.0]), g = 9.81, rho = 1000, n_Maning = 0.02):
    # Renvoie la matrice Jacobienne du terme Source par rapport au vecteur des variable d'état
    #Ui : vecteur des variables d'états à l'instant présent
    #z : vecteur des dérivés du fond (supposé linéaire ici)
    #g : accélération de la pesanteur

    if (Ui[0] > 1e-5) or (Ui[1]**2 + Ui[2]**2 > 1e-5) :
        hi, ui, vi = Ui[0], Ui[1]/Ui[0], Ui[2]/Ui[0]
    else:
        hi, ui, vi = Ui[0], 0, 0

    dU = np.array([[0, 0, 0],
                   [-g*z[0] + (10/3)*g*n_Maning*n_Maning*ui*((np.sqrt((ui)**2 + (vi)**2))/(np.cbrt(hi**7))), -g*n_Maning*n_Maning*((2*(ui)**2 + (vi)**2)/(np.cbrt(hi**7)*np.sqrt((ui)**2 + (vi)**2))), -g*n_Maning*n_Maning*((ui*vi)/(np.cbrt(hi**7)*np.sqrt((ui)**2 + (vi)**2)))], 
                   [-g*z[1] + (10/3)*g*n_Maning*n_Maning*vi*((np.sqrt((ui)**2 + (vi)**2))/(np.cbrt(hi**7))), -g*n_Maning*n_Maning*((ui*vi)/(np.cbrt(hi**7)*np.sqrt((ui)**2 + (vi)**2))), -g*n_Maning*n_Maning*(((ui)**2 + 2*(vi)**2)/(np.cbrt(hi**7)*np.sqrt((ui)**2 + (vi)**2)))]])

    return dU

def JacFlux(Ui, Uj, n, g = 9.81):
    # Renvoie la matrice Jacobienne du terme Flux par rapport au vecteur des variable d'état
    #Ui : vecteur des variables d'états à l'instant présent

    if Ui[0] > 1e-5:
        hi, ui, vi = Ui[0], Ui[1]/Ui[0], Ui[2]/Ui[0]
        hj, uj, vj = Uj[0], Uj[1]/Uj[0], Uj[2]/Uj[0]
    else:
        hi, ui, vi = Ui[0], 0, 0
        hj, uj, vj = Uj[0], 0, 0

    Lambda = LambdaMax(Ui, Uj.flatten(), n)
    dfi = n[0]*np.array([[0, 1, 0], [g*hi - ui*ui, 2*ui, 0], [-ui*vi, vi, ui]]) + n[1]*np.array([[0, 0, 1], [-ui*vi, vi, ui], [-g*hi - vi*vi, 0, 2*vi]])
    dfj = n[0]*np.array([[0, 1, 0], [g*hj - uj*uj, 2*uj, 0], [-uj*vj, vj, uj]]) + n[1]*np.array([[0, 0, 1], [-uj*vj, vj, uj], [-g*hj - vj*vj, 0, 2*vj]])
    dFi, dFj = (1/2)*(dfi + Lambda*np.eye(3)), (1/2)*(dfj - Lambda*np.eye(3))

    return dFi, dFj

def ImpGlo_A_R(coo, T, Un, Uguess, dt, Airs, Voisins, Normals):
    # Génère la matrice A et la vecteur de Reste de la résolution implicite globale
    #coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage
    #Un : vecteur des variables d'états à l'instant présent
    #Uguess : vecteur d'approximation des variables d'états à l'instant suivant
    #dt : pas de temps
    #Air : liste qui contient les airs des triangles donnés
    #Voisins : liste qui contient le triangle associé à chaque arête
    #Normals : liste des vecteurs normaux de chaque arête

    num_tri, num_var = Uguess.shape[0], Uguess.shape[1]
    N = num_var*num_tri
    A_global = lil_matrix((N, N))
    R_global = np.zeros((num_tri, num_var))

    for i, tri in enumerate(T):
        s1, s2, s3 = tri
        A = np.array(coo[s1])
        B = np.array(coo[s2])
        C = np.array(coo[s3])

        idx_tri = slice(i *3, (1 + i)*3)
        Long = LongArete(A, B, C)
        Air_T, Norma_T, Voi_T, Val_T = Airs[i], Normals[i], Voisins[i], Uguess[i]
        q = Val_T[1]**2 + Val_T[2]**2 
        if q < 1e-12 : #Précaution pour éviter les NaN dans le cas initiale avec u=v=0 et les cas ou la vitesse est presque nulle.
            dS = np.zeros((3, 3))
        else:
            dS = Jaco_S(Val_T)
        Aii= (Air_T/dt)*np.eye(3) - Air_T*dS
        A_global[idx_tri, idx_tri] += Aii
        Source_T = TriSource(np.array([0]), Val_T[0].reshape(1, -1), Val_T[1].reshape(1, -1), Val_T[2].reshape(1, -1))
        Ri = (Air_T/dt)*(Uguess[i] - Un[i]) - Air_T*Source_T[0].flatten()
        R_global[i] += Ri
        for idx, j in enumerate(Voi_T):
            if j[0] != None :
                idx_voi = slice(j[0] *3, (1 + j[0])*3)
                nij = Norma_T[idx]
                Val_Voi = Uguess[j[0]]

                dFi, dFj = JacFlux(Val_T, Val_Voi, nij)
                A_global[idx_tri, idx_tri] += Long[idx]*dFi
                A_global[idx_tri, idx_voi] += Long[idx]*dFj
                L_max = LambdaMax(Val_T, Val_Voi.flatten(), nij)
                Rusanov = 0.5*(Flux(Val_T, nij) + Flux(Val_Voi.flatten(), nij)) - 0.5*L_max*(Val_Voi.flatten() - Val_T)
                R_global[i] += Long[idx]*Rusanov

            else:
                nij = Norma_T[idx]
                Val_Voi = EtatMirroir(Val_T, nij)
                L_max = LambdaMax(Val_T, Val_Voi.flatten(), nij)
                Rusanov = 0.5*(Flux(Val_T, nij) + Flux(Val_Voi.flatten(), nij)) - 0.5*L_max*(Val_Voi.flatten() - Val_T)
                R_global[i] += Long[idx]*Rusanov

    return A_global, -R_global.flatten()

def ResolutionImpGlo(Coo, T, Uinit, dt, Time, Z_fond, iter_max):
    #Renvoit la résolution implicite globale des variables d'états et l'energie pour tout instant dt entre 0 et le temps final Time
    #Coo : vecteur contenant les coordonnées des noeuds du maillage
    #T : liste des Triangles du maillage
    #Uinit : vecteur des valeurs moyennes des variables d'états initiales pour chaque triangle
    #dt : pas de temps
    #Time : limite de temps finale
    #Z_fond : Liste des valeurs du fond pour chaque triangle
    #iter_max : nombe d'itération maximal pour le point fixe de la méthode implicite

    H, Hu, Hv = Uinit[0], Uinit[0]*Uinit[1], Uinit[0]*Uinit[2]
    U_now = np.vstack((H, Hu, Hv)).T
    Tri_air = Geo.AirTri(Coo, T)
    Voisins = Geo.Voisinage(T)
    Normals = NormalVec(Coo, T)

    Historia = []
    Historia.append(U_now)
    EnergieTot = []
    E0 = GloEnergie(U_now, Tri_air)
    EnergieTot.append(np.sum(E0))
    for t in tqdm(np.arange(0, Time+dt, dt)):
        U_old = U_now.copy() # Valeur à t (fixée pour cette itération)
        U_guess = U_now.copy() # Première estimation, U^{(k)}
        for k in range(iter_max):
            Energie_t = []
            A, R = ImpGlo_A_R(Coo, T, U_old, U_guess, dt, Tri_air, Voisins, Normals)
            Delta_U = spsolve(csr_matrix(A), -R) # Résolution du système linéaire A · dU = -R
            U_guess += Delta_U.reshape((-1, 3))

            if np.linalg.norm(Delta_U, ord=np.inf) < 1e-10:
                break

        U_now[:] = U_guess.copy()
        Historia.append(U_guess)
        Energie_t = GloEnergie(U_now, Tri_air)
        EnergieTot.append(np.sum(Energie_t))

    EnergieTot = np.array(EnergieTot)

    return Historia, EnergieTot

# ===========================================================================================================================================
# ===========================================================================================================================================
# ===========================================================================================================================================
# ===========================================================================================================================================

def MailleInterpo(Sol, T, Num):
    # Renvoit l'interpolation sur la maillage des valeurs obtenues par volumes finis
    #Sol : Vecteur de valeur par volume
    #T : liste des Triangles du maillage
    #Num : nombre de noeud du maillage

    add_H = np.zeros(Num)
    add_U = np.zeros(Num)
    add_V = np.zeros(Num)
    Compte = np.zeros(Num)
    for tri_idx, tri in enumerate(T):
        h, hu, hv = Sol[tri_idx]
        u, v = hu/h, hv/h
        for noeud in tri:
            add_H[noeud] += h
            add_U[noeud] += u
            add_V[noeud] += v
            Compte[noeud] += 1

    # Éviter division par zéro
    filtre = Compte > 0
    new_H = np.zeros(Num)
    new_U = np.zeros(Num)
    new_V = np.zeros(Num)

    new_H[filtre] = add_H[filtre] / Compte[filtre]
    new_U[filtre] = add_U[filtre] / Compte[filtre]
    new_V[filtre] = add_V[filtre] / Compte[filtre]

    return np.stack([new_H, new_U, new_V], axis=1)
 
