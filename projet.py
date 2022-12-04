# -*- coding: utf-8 -*-
# author : Augustin Cobena

from msilib.schema import Error
from tkinter import E
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
import time

import solutions
import solutions2
import fonctions

def generate_helmholtz():
    # Fréquence de résonance (calculée) : 77 Hz

    xmin, xmax, ymin, ymax = 0.0, 2.0, 0.0, 1.0
    nelemsx,nelemsy = 32,16

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    barycenters = fonctions.compute_barycenter(node_coords, p_elem2nodes, elem2nodes)
    node_coords, p_elem2nodes, elem2nodes = fonctions.remove_area(node_coords, p_elem2nodes, elem2nodes,0.751,0.874,0,0.437)
    node_coords, p_elem2nodes, elem2nodes = fonctions.remove_area(node_coords, p_elem2nodes, elem2nodes,0.751,0.874,0.563,1)
    node_coords, p_elem2nodes, elem2nodes = fonctions.remove_area(node_coords, p_elem2nodes, elem2nodes,0.873,2,0,0.249)
    node_coords, p_elem2nodes, elem2nodes = fonctions.remove_area(node_coords, p_elem2nodes, elem2nodes,0.873,2,0.751,1)
    node_coords, p_elem2nodes, elem2nodes = fonctions.remove_area(node_coords, p_elem2nodes, elem2nodes,1.376,2,0,1)

    # on fait le ménage des points inutiles
    for nodeid in reversed(range(node_coords.shape[0])): # liste en reversed pour renuméroter un minimum de points à chaque fois qu'on doit appeler remove_node
        if not nodeid in elem2nodes:
            node_coords,p_elem2nodes,elem2nodes = fonctions.remove_node_from_mesh(node_coords,p_elem2nodes,elem2nodes,nodeid,check_elems=False)

    wall_boundary = []
    boundary = fonctions.find_border_nodes(node_coords, p_elem2nodes, elem2nodes)
    for node in boundary:
        if not (node_coords[node][0] == 0 or node_coords[node][1] == 0 or node_coords[node][1] == 1):
            wall_boundary.append(node)
    wall_boundary = numpy.array(wall_boundary)

    node_coords, p_elem2nodes, elem2nodes = fonctions.transform_quad_to_tri(node_coords, p_elem2nodes, elem2nodes)
    nelems = p_elem2nodes.shape[0] - 1
    nnodes = node_coords.shape[0]

    return node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary


def generate_fractal(level):
    # Quelques fréquences intéressantes
    # Pas du tout localisées: 2817
    # Très localisées: 82, 138, 208, 265, 1002
    # Localisées + pertes d'énergie: 733, 3824
    # Pas localisées et peu de pertes d'énergies: 295, 3810

    node_coords, p_elem2nodes, elem2nodes = fonctions.generate_fractal(level)

    boundary = fonctions.find_border_nodes(node_coords, p_elem2nodes, elem2nodes)
    wall_boundary = []
    for node in boundary:
        if not (node_coords[node][0] == 0 or node_coords[node][1] == 0 or node_coords[node][1] == 1):
            wall_boundary.append(node)
    wall_boundary = numpy.array(wall_boundary)

    node_coords, p_elem2nodes, elem2nodes = fonctions.transform_quad_to_tri(node_coords, p_elem2nodes, elem2nodes)
    nelems = p_elem2nodes.shape[0] - 1
    nnodes = node_coords.shape[0]

    return node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary


def resolve(frequency,level = 2,mesh = None,generation="fractal", f = None):

    c = 340
    wavenumber = 2*numpy.pi*frequency/c

    if mesh == None:
        if generation == "fractal":
            node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary = generate_fractal(level)
        else:
            node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary = generate_helmholtz()
    else:
        node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary = mesh

    try:
        f[0]
    except:
        f = numpy.ones((nnodes,1))

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions2._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    nodes_dirichlet_boundary = []

    for nodeid in range(nnodes):
        if node_coords[nodeid][0] == 0 :
            nodes_dirichlet_boundary.append(nodeid)
    nodes_dirichlet_boundary = numpy.array(nodes_dirichlet_boundary)
    values_dirichlet_boundary = numpy.zeros(nnodes)

    # -- apply Dirichlet boundary conditions
    A, B = solutions2._set_dirichlet_condition(nodes_dirichlet_boundary, values_dirichlet_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)
    solreal = sol.reshape((sol.shape[0], ))

    if mesh == None:
        _ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))

    return solreal

def existence_surface(sol):
    s = (numpy.linalg.norm(sol,ord=2)**2)/(numpy.linalg.norm(sol,ord=4)**4)
    return s

def energy_losses(sol,boundary):
    e = (numpy.linalg.norm(sol,ord=2)**2)/(numpy.linalg.norm(sol[boundary],ord=2)**2)
    return e

def find_localizing_frequencies(level = 2,bandwidth = [x for x in range(20,900)]):
    beginning = time.time()
    print("Calculating all frequencies...")

    node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary = generate_fractal(level)

    max = -numpy.inf
    max_frequency = None
    solmax = None

    values = []
    for frequency in bandwidth:
        sol = resolve(frequency,level,mesh=(node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary))
        s = existence_surface(sol)
        e = energy_losses(sol,wall_boundary)

        value_to_maximize = e / s
        if value_to_maximize > max:
            max = value_to_maximize
            max_frequency = frequency
            solmax = sol
        values.append(value_to_maximize)
    
    end = time.time()
    
    print("Done")
    print("Temps écoulé:",end - beginning)
    matplotlib.pyplot.plot(bandwidth, values)
    matplotlib.pyplot.yscale("log")
    matplotlib.pyplot.show()
    print("Max:",max)
    print("Fréquence associée:",max_frequency)
    _ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solmax))

def find_eigenvalues(frequency,level = 2, generation = "fractal",display = True):
    c = 340
    wavenumber = 2*numpy.pi*frequency/c

    if generation == "fractal":
        node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary = generate_fractal(level)
    else:
        node_coords, p_elem2nodes, elem2nodes, nelems, nnodes, wall_boundary = generate_helmholtz()

    # f = 100*numpy.random.rand(nnodes,1)
    f = numpy.ones((nnodes,1))
    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions2._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    nodes_dirichlet_boundary = []

    for nodeid in range(nnodes):
        if node_coords[nodeid][0] == 0 :
            nodes_dirichlet_boundary.append(nodeid)
    nodes_dirichlet_boundary = numpy.array(nodes_dirichlet_boundary)
    values_dirichlet_boundary = numpy.zeros(nnodes)

    # -- apply Dirichlet boundary conditions
    A, B = solutions2._set_dirichlet_condition(nodes_dirichlet_boundary, values_dirichlet_boundary, A, B)

    W,V = numpy.linalg.eig(A)
    
    if display:
        S = [V[:,j] for j in range(nnodes)]
        S = sorted(S,key=lambda x: energy_losses(x,wall_boundary)/existence_surface(x))
        for j in range(3):
            _ = solutions2._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(S[j]))
    
    return S[0]



if __name__ == "__main__":

    # Résolution géométrie Helmholtz

    resolve(77,generation="helmholtz")
    resolve(200,generation="helmholtz")

    find_eigenvalues(0,generation="helmholtz")

    # Résolution géométrie fractale

    resolve(138,generation="fractal")
    resolve(450,generation="fractal")
    
    find_localizing_frequencies(bandwidth=[x for x in range(20,900)])

    find_eigenvalues(0,generation="fractal")

    # Vecteur propre en tant que f
    frequency = 300
    resolve(frequency)
    f = find_eigenvalues(frequency,display=False)
    resolve(frequency,f=f)