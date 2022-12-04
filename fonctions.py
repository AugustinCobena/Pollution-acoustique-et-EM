# -*- coding: utf-8 -*-
# author : Augustin Cobena

from msilib.schema import Error
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
import random

import solutions
import solutions2

# -------------------------------------------------------------------    Fonctions utiles pour les calculs    ----------------------------------------------------------------------

def length(a,b):
    # Longueur d'un segment, prend des coordonnées en entrée
    return numpy.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def unit_vector(a,b):
    # Vecteur unitaire de A vers B, prend des coordonnées en entrée, renvoie des coordonnées
    l = length(a,b)
    return((b - a)/l)

def scalar_product(u,v):
    # Produit scalaire de 2 vecteurs
    return(u[0]*v[0] + u[1]*v[1] + u[2]*v[2])

def line_intersection(a,b,c,d):
    # Prend 4 coordonnées de points en entrée, renvoie les coordonnées de l'intersection de (AB) et (CD)
    xdiff = (a[0] - b[0], c[0] - d[0])
    ydiff = (a[1] - b[1], c[1] - d[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(a,b), det(c,d))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return numpy.array([x,y,0])

def area_tri(x,y,z):
    # Calcule l'aire d'un triangle avec les coordonnées des points, selon une formule qui utilise le rayon du cercle inscrit

    a = length(x,y)
    b = length(y,z)
    c = length(z,x)

    s = (a+b+c)/2
    rho = numpy.sqrt((s-a)*(s-b)*(s-c)/s) # rayon du cercle inscrit

    area = rho*s

    return area

# --------------------------------------------------------------    Fonctions utiles pour la gestion d'un maillage   ----------------------------------------------------------------

def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, this_node_coords):
    # Ajoute un noeud
    node_coords = numpy.append(node_coords,[this_node_coords],axis=0)
    return node_coords, p_elem2nodes, elem2nodes


def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodes):
    # Ajoute un élément
    for node in nodes:
        elem2nodes = numpy.append(elem2nodes,node)
    p_elem2nodes = numpy.append(p_elem2nodes,p_elem2nodes[-1] + nodes.shape[0])
    return node_coords, p_elem2nodes, elem2nodes


def remove_node_from_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid, check_elems = True):
    # Supprime un noeud ainsi que les éléments associés, renumérote les noeuds suivants
    node_coords = numpy.delete(node_coords,nodeid,axis = 0)

    if check_elems:
        to_remove = []
        for elemid in range(p_elem2nodes.shape[0] - 1):
            beginning = p_elem2nodes[elemid]
            end = p_elem2nodes[elemid + 1]
            if nodeid in elem2nodes[beginning:end]:
                to_remove.insert(0,elemid) # /!\ attention il est important d'insert au debut car il faut que la liste soit triée dans l'ordre décroissant pour la suite
        
        for elemid in to_remove:
            node_coords, p_elem2nodes, elem2nodes = remove_elem_from_mesh(node_coords, p_elem2nodes, elem2nodes, elemid)
    
    elem2nodes[elem2nodes > nodeid] -= 1

    return node_coords, p_elem2nodes, elem2nodes,


def remove_elem_from_mesh(node_coords, p_elem2nodes, elem2nodes, elemid):
    # Supprime un élément, renumérote les indices suivants dans p_elem2nodes
    beginning = p_elem2nodes[elemid]
    end = p_elem2nodes[elemid + 1]

    elem2nodes = numpy.delete(elem2nodes,range(beginning,end))

    p_elem2nodes = numpy.delete(p_elem2nodes,elemid)
    p_elem2nodes[elemid:] -= (end - beginning)

    return node_coords, p_elem2nodes, elem2nodes,

def remove_area(node_coords, p_elem2nodes, elem2nodes, xmin, xmax, ymin, ymax,barycenter_elems=False):
    # Supprime les noeuds situés strictement à l'intérieur du rectangle de coordonnées données en entrée, ainsi que les éléments dont le barycentre est à l'intérieur
    if not barycenter_elems:
        barycenter_elems = compute_barycenter(node_coords,p_elem2nodes,elem2nodes)

    to_remove = []
    for elemid in range(p_elem2nodes.shape[0] - 1):
        if barycenter_elems[elemid][0]  > xmin and barycenter_elems[elemid][0] < xmax and barycenter_elems[elemid][1] > ymin and barycenter_elems[elemid][1] < ymax:
            to_remove.insert(0, elemid) # /!\ attention il est important d'insert au debut car il faut que la liste soit triée dans l'ordre décroissant pour la suite
    
    for elemid in to_remove:
        node_coords, p_elem2nodes, elem2nodes = remove_elem_from_mesh(node_coords, p_elem2nodes, elem2nodes, elemid)
    
    to_remove = []
    for nodeid in range(node_coords.shape[0]):
        if node_coords[nodeid][0] > xmin and node_coords[nodeid][0] < xmax and node_coords[nodeid][1] > ymin and node_coords[nodeid][1] < ymax:
            to_remove.insert(0,nodeid)
    
    for nodeid in to_remove:
        node_coords, p_elem2nodes, elem2nodes = remove_node_from_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid)

    return node_coords, p_elem2nodes, elem2nodes

def randomize(node_coords, p_elem2nodes, elem2nodes, delta_x, delta_y):
    # Ajoute de l'aléatoire dans un maillage rectangulaire
    for node_id in range(node_coords.shape[0]):
        # on ne bouge que les points qui ne sont pas au bord
        if node_coords[node_id][0] != 0 and node_coords[node_id][0] != 1 and node_coords[node_id][1] != 0 and node_coords[node_id][1] != 1:
            node_coords[node_id][0] += random.uniform(-delta_x,delta_x)
            node_coords[node_id][1] += random.uniform(-delta_y,delta_y)
    return node_coords, p_elem2nodes, elem2nodes

def generate_fractal(level):
    # Génère une fractale sur le coté droit. L'aire totale sera toujours égale à 1
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 2.0
    
    nelemsx, nelemsy = 4**level , 2 * 4**level
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)

    mask_matrix = numpy.empty((nelemsx,nelemsy))
    mask_matrix[:,:nelemsy//2] = 1
    mask_matrix[:,nelemsy//2:] = 0
    
    
    def fractal_recursion(i,j,k,l,level):
        if level > 0:
            level -= 1

            x = (k - i)//4
            y = (l - j)//4

            # si le segment est vertical, y et nul. si le segment est horizontal, x est nul. On utilise le même code pour simplifier

            # On construit les points de cette partie de la fractale
            p = [0 for z in range(9)]
            p[0] = ( i , j )
            p[1] = ( i+x , j+y )
            p[2] = ( i+x-y , j+x+y )
            p[3] = ( i+2*x-y , j+x+2*y )
            p[4] = ( i+2*x , j+2*y )
            p[5] = ( i+2*x+y , j-x+2*y )
            p[6] = ( i+3*x+y , j-x+3*y )
            p[7] = ( i+3*x , j+3*y )
            p[8] = ( k , l )

            # On inverse les points dans la fractale
            imin = min(p[1][0],p[3][0])
            imax = max(p[1][0],p[3][0])
            jmin = min(p[1][1],p[3][1])
            jmax = max(p[1][1],p[3][1])

            mask_matrix[imin:imax,jmin:jmax] = (mask_matrix[imin:imax,jmin:jmax] + 1) % 2

            imin = min(p[5][0],p[7][0])
            imax = max(p[5][0],p[7][0])
            jmin = min(p[5][1],p[7][1])
            jmax = max(p[5][1],p[7][1])

            mask_matrix[imin:imax,jmin:jmax] = (mask_matrix[imin:imax,jmin:jmax] + 1) % 2
            
            # On appelle la fonction récursive de construction sur tous les segments du bout de fractale qu'on vient de créer
            for i in range(8):
                fractal_recursion(p[i][0],p[i][1],p[i+1][0],p[i+1][1],level)
    
    fractal_recursion(0,nelemsx,nelemsx,nelemsx,level)

    # Dans un tableau, le premier indice est celui des lignes et le 2e celui des colonnes.
    # Pour que les indices de ma matrice correspondent au mesh, je met l'axe des x vertical et celui des y horizontal.
    # (C'est simplement pour pouvoir faire la correspondance facilement entre la matrice et le maillage pour le debugging)
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(ymin, ymax, xmin, xmax, nelemsy, nelemsx)

    for x in reversed(range(nelemsx)):
        for y in reversed(range(nelemsy)):
            if not mask_matrix[x,y]:
                node_coords,p_elem2nodes,elem2nodes = remove_elem_from_mesh(node_coords,p_elem2nodes,elem2nodes,elemid=y + x*nelemsy)

    # on fait le ménage des points inutiles
    for nodeid in reversed(range(node_coords.shape[0])): # liste en reversed pour renuméroter un minimum de points à chaque fois qu'on doit appeler remove_node
        if not nodeid in elem2nodes:
            node_coords,p_elem2nodes,elem2nodes = remove_node_from_mesh(node_coords,p_elem2nodes,elem2nodes,nodeid,check_elems=False)
    
    return node_coords, p_elem2nodes, elem2nodes

def transform_quad_to_tri(node_coords,p_elem2nodes,elem2nodes):
    # Coupe les quadrangles en 2 pour former 2 triangles
    nelems = p_elem2nodes.shape[0] - 1

    new_p_elem2nodes = numpy.empty(nelems*2+1, dtype= numpy.int64)
    new_elem2nodes = numpy.empty(nelems*6, dtype=numpy.int64)

    for elemid in range(nelems):

        beginning = p_elem2nodes[elemid]
        end = p_elem2nodes[elemid + 1]
        nodes = elem2nodes[beginning:end]

        if end - beginning != 4:
            raise Exception("pas un quadrangle")

        new_elem2nodes[elemid * 6 : elemid * 6 + 3] = nodes[:3]

        new_elem2nodes[elemid * 6 + 3] = nodes[0]
        new_elem2nodes[elemid * 6 + 4] = nodes[2]
        new_elem2nodes[elemid * 6 + 5] = nodes[3]

        new_p_elem2nodes[elemid*2:elemid*2+2] = [elemid * 6 , elemid * 6 + 3]
    
    new_p_elem2nodes[-1] = nelems * 6 + 5

    return node_coords, new_p_elem2nodes, new_elem2nodes

def find_border(node_coords, p_elem2nodes, elem2nodes):
    # Renvoie un tableau d'éléments à 2 noeuds qui constituent le bord du maillage. Je préfère le construire en dehors de elem2nodes tout en conservant une structure qui permet
    # de le rajouter très facilement à elem2nodes
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)
    nelems = p_elem2nodes.shape[0] - 1
    border = []

    for elemid in range(nelems):
        if p_elem2nodes[elemid + 1] - p_elem2nodes[elemid] != 4:
            raise Exception("pas des quadrangles")

        nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        c0 = p_node2elems[nodes[0]+1] - p_node2elems[nodes[0]]
        c1 = p_node2elems[nodes[1]+1] - p_node2elems[nodes[1]]
        c2 = p_node2elems[nodes[2]+1] - p_node2elems[nodes[2]]
        c3 = p_node2elems[nodes[3]+1] - p_node2elems[nodes[3]]
        c = numpy.array([c0,c1,c2,c3])
        
        i = 0
        # On regarde toutes les arêtes de l'élément. Le premier filtre (pour réduire le temps de calcul) est simplement de voir si les 2 noeuds appartiennent à moins de 4 éléments
        # Le 2e filtre (nécessaire et suffisant) est de vérifier que l'arête n'appartient qu'à 1 élément, c'est à dire que les 2 noeuds n'ont qu'un seul élément en commun
        while i < 4:
            if c[i] < 4:
                if c[(i+1)%4] < 4:
                    elems1 = node2elems[p_node2elems[nodes[i]]:p_node2elems[nodes[i] + 1]]
                    elems2 = node2elems[p_node2elems[nodes[(i+1)%4]]:p_node2elems[nodes[(i+1)%4] + 1]]
                    if len(numpy.intersect1d(elems1,elems2)) <= 1:
                        border.append(nodes[i])
                        border.append(nodes[(i+1)%4])
            i += 1

    return numpy.array(border)

            
def find_border_nodes(node_coords, p_elem2nodes, elem2nodes):
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)
    nnodes = p_node2elems.shape[0] - 1
    border = []

    for nodeid in range(nnodes):
        if (p_node2elems[nodeid + 1] - p_node2elems[nodeid]) < 4:
            border.append(nodeid)
    
    return numpy.array(border)


# ----------------------------------------------------------------    Fonctions de cacul de paramètres divers    ---------------------------------------------------------------------

def build_node2elems(p_elem2nodes, elem2nodes):
    # Construit la matrice node2elems, la renvoie sous la forme d'une matrice presque vide compressée
    # elem2nodes connectivity matrix
    e2n_coef = numpy.ones(len(elem2nodes), dtype=numpy.int64)
    e2n_mtx = scipy.sparse.csr_matrix((e2n_coef, elem2nodes, p_elem2nodes))
    # node2elems connectivity matrix
    n2e_mtx = e2n_mtx.transpose()
    n2e_mtx = n2e_mtx.tocsr()
    # output
    p_node2elems = n2e_mtx.indptr
    node2elems = n2e_mtx.indices

    return p_node2elems, node2elems

def compute_h(node_coords, p_elem2nodes, elem2nodes,method="radius_ratio"):
    # Calcule h avec méthode au choix: avg, max ou radius_ratio
    spacedim = node_coords.shape[1]
    nelems = p_elem2nodes.shape[0] - 1
    h = numpy.zeros((nelems,spacedim))

    for elemid in [0]:
        nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        if nodes.shape[0] != 3:
            raise Exception("not a triangle")

        a = length(node_coords[nodes[0]],node_coords[nodes[1]])
        b = length(node_coords[nodes[1]],node_coords[nodes[2]])
        c = length(node_coords[nodes[2]],node_coords[nodes[0]])

        if method == "avg":
            h[elemid] = (a+b+c)/3

        if method == "max":
            h[elemid] = max(a,b,c)
        
        if method == "radius_ratio":
            s = (a+b+c)/2
            incircle_radius = numpy.sqrt((s-a)*(s-b)*(s-c)/s)
            circumcircle_radius = a*b*c/(2*numpy.sqrt(s*(s-a)*(s-b)*(s-c)))

            h[elemid] = numpy.sqrt(incircle_radius*circumcircle_radius)
    
    return numpy.average(h)

def compute_barycenter(node_coords, p_elem2nodes, elem2nodes):
    # Renvoie un tableau du barycentre de tous les éléments
    spacedim = node_coords.shape[1]
    nelems = p_elem2nodes.shape[0] - 1
    elem_coords = numpy.zeros((nelems,spacedim))

    for elem_number in range(nelems):

        beginning = p_elem2nodes[elem_number]
        end = p_elem2nodes[elem_number + 1]
        barycenter = numpy.zeros(spacedim)

        for node_id in elem2nodes[beginning:end]:
            barycenter += node_coords[node_id]
        barycenter = barycenter/(end - beginning)

        elem_coords[elem_number] = barycenter

    return elem_coords

def compute_aspect_ratio_of_tri(node_coords, p_elem2nodes, elem2nodes):
    # Calcul l'aspect ratio d'un triangle selon la formule vue en cours
    nelems = p_elem2nodes.shape[0] - 1
    tri_quality = numpy.zeros(nelems)
    for elemid in range(nelems):
        nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        if nodes.shape[0] != 3:
            raise Exception("pas un triangle")
        
        alpha = numpy.sqrt(3)/6

        a = length(node_coords[nodes[0]],node_coords[nodes[1]])
        b = length(node_coords[nodes[1]],node_coords[nodes[2]])
        c = length(node_coords[nodes[2]],node_coords[nodes[0]])

        hmax = max(a,b,c)

        s = (a+b+c)/2
        rho = numpy.sqrt((s-a)*(s-b)*(s-c)/s) # formule du rayon du cercle inscrit

        tri_quality[elemid] = alpha * hmax / rho

    return tri_quality

def compute_aspect_ratio_of_quad(node_coords, p_elem2nodes, elem2nodes):
    # Calcule l'aspect ratio d'un quadrilatère selon la formule vue en cours
    nelems = p_elem2nodes.shape[0] - 1
    quad_quality = numpy.zeros(nelems)
    for elemid in range(nelems):
        nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        if nodes.shape[0] != 4:
            raise Exception("pas un quadrilatère")
        e1 = unit_vector(node_coords[nodes[0]],node_coords[nodes[1]])
        e2 = unit_vector(node_coords[nodes[1]],node_coords[nodes[2]])
        e3 = unit_vector(node_coords[nodes[2]],node_coords[nodes[3]])
        e4 = unit_vector(node_coords[nodes[3]],node_coords[nodes[0]])
        quad_quality[elemid] = 1 - (abs(scalar_product(e1,e2)) + abs(scalar_product(e2,e3)) + abs(scalar_product(e3,e4)) + abs(scalar_product(e4,e1)))/4
    return quad_quality

def compute_edge_length_factor_of_tri(node_coords, p_elem2nodes, elem2nodes):
    # Calcule le edge length factor d'un triangle selon la formule vue en cours
    nelems = p_elem2nodes.shape[0] - 1
    tri_edge_length_factor = numpy.zeros(nelems)
    for elemid in range(nelems):
        nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        if nodes.shape[0] != 3:
            raise Exception("pas un triangle")

        a = length(node_coords[nodes[0]],node_coords[nodes[1]])
        b = length(node_coords[nodes[1]],node_coords[nodes[2]])
        c = length(node_coords[nodes[2]],node_coords[nodes[0]])

        s = min(a,b,c)
        if a == s:
            if b < c:
                m = b
            else:
                m = c
        elif b == s:
            if a < c:
                m = a
            else:
                m = b
        else:
            if a < b:
                m = a
            else:
                m = b
        
        tri_edge_length_factor[elemid] = s/m

    return tri_edge_length_factor


def compute_pointedness_of_quad(node_coords, p_elem2nodes, elem2nodes):
    # Calcule la pointedness d'un quadrilatère selon la formule vue en cours
    nelems = p_elem2nodes.shape[0] - 1
    quad_quality = numpy.zeros(nelems)
    for elemid in range(nelems):
        nodes = elem2nodes[p_elem2nodes[elemid]:p_elem2nodes[elemid + 1]]
        if nodes.shape[0] != 4:
            raise Exception("pas un quadrilatère")
        a = node_coords[nodes[0]]
        b = node_coords[nodes[1]]
        c = node_coords[nodes[2]]
        d = node_coords[nodes[3]]

        r = line_intersection(a,c,b,d)

        A1 = area_tri(a,b,r)
        A2 = area_tri(b,c,r)
        A3 = area_tri(c,d,r)
        A4 = area_tri(d,a,r)

        Amin = min(A1,A2,A3,A4)
        A = A1+A2+A3+A4

        quad_quality[elemid] = 4*Amin/A
    
    return quad_quality

def compute_error(size=20,wavenumber=numpy.pi,randomized=False,method=None):
    # Calcule l'erreur entre solution exacte et simulation pour un maillage carré simple

    # -- set equation parameters
    # wavenumber = numpy.pi
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = size,size
    delta_x = (xmax - xmin)/(3*nelemsx)
    delta_y = (ymax - ymin)/(3*nelemsy)

    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions2._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    if randomized:
        node_coords, p_elem2nodes, elem2nodes = randomize(node_coords, p_elem2nodes, elem2nodes, delta_x, delta_y)
    #node_coords, p_elem2nodes, elem2nodes = randomize(node_coords, p_elem2nodes, elem2nodes, delta_x, delta_y)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # compute h (only if we need it)
    h = None
    if method:
        h = compute_h(node_coords, p_elem2nodes, elem2nodes,method=method)

    # -- plot mesh
    # fig = matplotlib.pyplot.figure(1)
    # ax = matplotlib.pyplot.subplot(1, 1, 1)
    # ax.set_aspect('equal')
    # ax.axis('off')
    # solutions2._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    # matplotlib.pyplot.show()

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = solutions2._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions2._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions2._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions2._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0.,1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions2._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions2._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))
    #_ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
    #_ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
    #
    # ..warning: for teaching purpose only
    # -- plot exact solution
    solexactreal = solexact.reshape((solexact.shape[0], ))
    #_ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
    #_ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
    # # ..warning: end

    # ..warning: for teaching purpose only
    # -- plot exact solution - approximate solution
    solerr = solreal - solexactreal
    #print(solerr)
    #_ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
    #_ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))
    # # ..warning: end

    return solerr,h

# -----------------------------------------------------------------------   Fonctions d'affichage   --------------------------------------------------------------------------------------

def plot_fractal(level=2):
    # Affiche une fractale ainsi que la bordure
    node_coords, p_elem2nodes, elem2nodes = generate_fractal(level)
    border = find_border(node_coords, p_elem2nodes, elem2nodes)
    border_nodes = find_border_nodes(node_coords, p_elem2nodes, elem2nodes)
    node_coords, p_elem2nodes, elem2nodes = transform_quad_to_tri(node_coords, p_elem2nodes, elem2nodes)

    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    for i in range(len(border)//2):
        matplotlib.pyplot.plot((node_coords[border[2*i]][0], node_coords[border[2*i+1]][0]), (node_coords[border[2*i]][1], node_coords[border[2*i+1]][1]), color="red")
    for node in border_nodes:
        matplotlib.pyplot.scatter(node_coords[node][0],node_coords[node][1],color="black")
    matplotlib.pyplot.show()

def plot_error_depending_on_h(wavenumber=numpy.pi,randomized=False,method="radius_ratio"):
    # Affiche l'erreur en fonction de h (échelle logarithmique). Méthode de calcul de h au choix.
    err = []
    h_list = []
    log_h = []
    log_err = []
    for size in [5,6,7,8,9,10,12,15,20,25,30]:
        solerr,h = compute_error(size=size,wavenumber=wavenumber,randomized=randomized,method=method)
        err.append(numpy.linalg.norm(solerr))
        h_list.append(h)
        log_h.append(numpy.log10(h))
        log_err.append(numpy.log10(numpy.linalg.norm(solerr)))

    alpha = numpy.polyfit(log_h,log_err,1)[0]

    matplotlib.pyplot.plot(h_list,err)
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.xlabel("h")
    matplotlib.pyplot.yscale("log")
    matplotlib.pyplot.ylabel("Erreur")
    matplotlib.pyplot.text(0.0001,0.1,"Alpha = " + str(alpha))
    matplotlib.pyplot.show()

def plot_error_depending_on_k(size=20,randomized=False,method="radius_ratio"):
    # Affiche l'erreur en fonction de k (échelle logarithmique)
    size = 20
    err = []
    k_list = []
    log_err = []
    log_k = []

    for k in [0.5,1,1.5,2,2.5,3,3.5]:
        solerr = compute_error(size=size,wavenumber=k,randomized=randomized,method=method)[0]
        err.append(numpy.linalg.norm(solerr))
        k_list.append(k)
        log_err.append(numpy.log(numpy.linalg.norm(solerr)))
        log_k.append(numpy.log(k))

    beta = numpy.polyfit(log_k,log_err,1)[0]

    matplotlib.pyplot.plot(k_list,err)
    matplotlib.pyplot.xscale("log")
    matplotlib.pyplot.xlabel("k")
    matplotlib.pyplot.yscale("log")
    matplotlib.pyplot.ylabel("Erreur")
    matplotlib.pyplot.text(0.6,0.01,"Beta = " + str(beta))
    matplotlib.pyplot.show()

def test_mesh_functions():
    # Permet de tester la correction des fonctions de gestion de maillage
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(0., 1., 0., 1., 10, 10)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_from_mesh(node_coords, p_elem2nodes, elem2nodes,3)
    node_coords, p_elem2nodes, elem2nodes = remove_area(node_coords, p_elem2nodes, elem2nodes, 0.3,0.7,0.2,0.5)
    node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, numpy.array([-0.12,0.15,0]))
    node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, numpy.array([-0.08,0.03,0]))
    node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes,numpy.array([0,11,115,116]))
    node_coords, p_elem2nodes, elem2nodes = transform_quad_to_tri(node_coords, p_elem2nodes, elem2nodes)
    node_coords, p_elem2nodes, elem2nodes = randomize(node_coords, p_elem2nodes, elem2nodes,delta_x=0.03,delta_y=0.03)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    matplotlib.pyplot.show()

def test_computation_functions():
    # Permet de tester la correction des fonctions de calcul

    delta_x = 0.03
    delta_y = 0.03

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(0., 1., 0., 1., 10, 10)
    barycenters = compute_barycenter(node_coords, p_elem2nodes, elem2nodes)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter([point[0] for point in barycenters],[point[1] for point in barycenters],s=5,color="red")
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')

    aspect_ratio = compute_aspect_ratio_of_quad(node_coords, p_elem2nodes, elem2nodes)
    print("Not randomized aspect ratio of quad")
    print("standard deviation:",numpy.std(aspect_ratio))
    print("mean:",numpy.mean(aspect_ratio))
    print("\n")

    pointedness = compute_pointedness_of_quad(node_coords, p_elem2nodes, elem2nodes)
    print("Not randomized pointedness of quad")
    print("standard deviation:",numpy.std(pointedness))
    print("mean:",numpy.mean(pointedness))
    print("\n")

    matplotlib.pyplot.show()

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(0., 1., 0., 1., 10, 10)
    node_coords, p_elem2nodes, elem2nodes = randomize(node_coords, p_elem2nodes, elem2nodes,delta_x,delta_y)
    barycenters = compute_barycenter(node_coords, p_elem2nodes, elem2nodes)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter([point[0] for point in barycenters],[point[1] for point in barycenters],s=5,color="red")
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')

    aspect_ratio = compute_aspect_ratio_of_quad(node_coords, p_elem2nodes, elem2nodes)
    print("Randomized aspect ratio of quad")
    print("standard deviation:",numpy.std(aspect_ratio))
    print("mean:",numpy.mean(aspect_ratio))
    print("\n")

    pointedness = compute_pointedness_of_quad(node_coords, p_elem2nodes, elem2nodes)
    print("Randomized pointedness of quad")
    print("standard deviation:",numpy.std(pointedness))
    print("mean:",numpy.mean(pointedness))  
    print("\n")

    matplotlib.pyplot.show()

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_trimesh(0., 1., 0., 1., 10, 10)
    barycenters = compute_barycenter(node_coords, p_elem2nodes, elem2nodes)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter([point[0] for point in barycenters],[point[1] for point in barycenters],s=5,color="red")
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')

    aspect_ratio = compute_aspect_ratio_of_tri(node_coords, p_elem2nodes, elem2nodes)
    print("Not randomized aspect ratio of tri")
    print("standard deviation:",numpy.std(aspect_ratio))
    print("mean:",numpy.mean(aspect_ratio))
    print("\n")

    edge_length_factor = compute_edge_length_factor_of_tri(node_coords, p_elem2nodes, elem2nodes)
    print("Not randomized edge length factor of tri")
    print("standard deviation:",numpy.std(edge_length_factor))
    print("mean:",numpy.mean(edge_length_factor))
    print("\n")

    matplotlib.pyplot.show()

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_trimesh(0., 1., 0., 1., 10, 10)
    node_coords, p_elem2nodes, elem2nodes = randomize(node_coords, p_elem2nodes, elem2nodes,delta_x,delta_y)
    barycenters = compute_barycenter(node_coords, p_elem2nodes, elem2nodes)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter([point[0] for point in barycenters],[point[1] for point in barycenters],s=5,color="red")
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')

    aspect_ratio = compute_aspect_ratio_of_tri(node_coords, p_elem2nodes, elem2nodes)
    print("Randomized aspect ratio of tri")
    print("standard deviation:",numpy.std(aspect_ratio))
    print("mean:",numpy.mean(aspect_ratio))
    print("\n")

    edge_length_factor = compute_edge_length_factor_of_tri(node_coords, p_elem2nodes, elem2nodes)
    print("Randomized edge length factor of tri")
    print("standard deviation:",numpy.std(edge_length_factor))
    print("mean:",numpy.mean(edge_length_factor))
    print("\n")

    matplotlib.pyplot.show()


if __name__ == '__main__':
    # Test génération fractale (teste aussi la génération de bordure)
    plot_fractal(level=2)

    # Test calculs alpha et beta
    plot_error_depending_on_h(wavenumber=numpy.pi,randomized=False,method="radius_ratio")
    plot_error_depending_on_k(size=20,randomized=False)

    # Test fonctions de gestion de maillage
    test_mesh_functions()

    # Test fonctions de calcul
    test_computation_functions()

    ()
