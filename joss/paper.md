---
title: 'Basix: a runtime finite element basis evaluation library'
tags:
  - Python
  - C++
  - finite element method
  - basis functions
  - numerical analysis
authors:
  - name: Chris N. Richardson
    orcid: 0000-0003-3137-1392
    affiliation: 1
  - name: Matthew W. Scroggs
    orcid: 0000-0002-4658-2443
    affiliation: 2
  - name: Garth N. Wells
    orcid: 0000-0001-5291-7951
    affiliation: 2
affiliations:
 - name: BP Institute, University of Cambridge
   index: 1
 - name: Department of Engineering, University of Cambridge
   index: 2
date: 26 October 2021
bibliography: paper.bib
---

# Summary

The finite element method (FEM) [@ciarlet] is a widely used numerical
method for approximating the solution of partial differential equations
(PDEs). Solving a problem using FEM involves discretising the problem
and searching for a solution in a finite dimensional space: these finite
spaces are created by defining a finite element on each cell of a mesh.

Following @ciarlet, a finite element is commonly defined by a triple
$(R, \mathcal{V}, \mathcal{L})$, where:

- $R$ is the reference cell, for example a triangle with vertices at
  (0,0), (1,0) and (0,1);
- $\mathcal{V}$ is a finite dimensional polynomial space, for example
  $\operatorname{span}\{1, x, y, x^2, xy, y^2\}$;
- $\mathcal{L}$ is a basis of the dual space
  $\{f:\mathcal{V}\to\mathbb{R}\}$, for example the set of functionals
  that evaluate a function at the vertices of the triangle and at the
  midpoints of its edges.

The basis functions of the finite element are the polynomials in
$\mathcal{V}$ such that one functional in $\mathcal{L}$ gives the value
1 for that function and all other functions in $\mathcal{L}$ give 0. The
examples given above define a degree 2 Lagrange space on a triangle; the
basis functions of this space are shown in \autoref{fig:fe}.

![The six basis functions of an order 2 Lagrange space on a triangle.
The uppper three functions arise from point evaluations at the vertices.
The lower three arise from point evaluations at the midpoints of the
edges. These diagrams are taken from DefElement
[@defelement].\label{fig:fe}](basis-functions.png){ width=60% }

The functionals in $\mathcal{L}$ are each associated with a degree of
freedom (DOF) of the finite element space. Each functional (or DOF) is
additionally associated with a sub-entity of the reference cell.
Ensuring that the same coefficients are assigned to the DOFs of
neighbouring cells associated with a shared sub-entity gives the finite
element space the desired continuity properties.

Basix is a C++ library that creates and tabulates a range finite
elements on triangles, tetrahedra, quadrilaterals, hexahedra, pyramids,
and prisms. Currently supported element types include Lagrange, Nédélec
first kind [@nedelec1], Nédélec second kind [@nedelec2], Raviart--Thomas
[@rt], Brezzi--Douglas--Marini [@bdm], Crouzeix--Raviart [@cr],
serendipity [@serendipity; @sdivcurl], and Regge [@regge; @regge2]
elements. The majority of Basix's functionality can be used via the
library's Python interface.

Basix forms part of FEniCSx alongside DOLFINx [@dolfinx], FFCx [@ffcx],
and UFL [@ufl]. FEniCSx is the latest development version of FEniCS, a
popular open source finite element project [@fenics].

# Statement of need

Basix allows users to:

- evaluate finite element basis functions and their derivatives at a set
  of points;
- access geometric and topological information about reference cells;
- apply push forward and pull back operations to map data between a
  reference cell and a physical cell;
- permute and transform DOFs to allow higher-order elements to be use on
  arbitrary meshes; and
- interpolate into a finite element space and between finite element
  spaces.

In many FEM libraries, the definitions of elements are included within
the code of the library rather then separating the element definition
and tabulation into a standalone library as we do. Following the latter
approach allows us to make adjustments to how elements are implemented
and add new elements to Basix without needing to make changes the rest
of the library. This also allows users who want to create custom
integration kernels to get information about elements from Basix without
having to extract information from the core of the full finite element
library.

The Python library FIAT [@fiat] (which is part of the legacy FEniCS
library alongside UFL, FFC [@ffc] and DOLFIN [@dolfin]) serves a
similiar purpose as Basix and can perform many of the same operatations
(with the exception of permutations and transformations) on triangles,
tetrahedra, quadrilaterals, and hexahedra. As FIAT is written in Python,
the FFC library would use the information from FIAT to generate code
that could be used by the C++ finite element library DOLFIN.

An advantage of using Basix is the ability to call functions from C++
at runtime. This has allowed us to greatly reduce the amount of code
generated in FFCx compared to FFC, as well as simplifying much of the
implementation, while still allowing FFCx to access the information it
needs using Basix's Python interface.

Another key advantage of Basix is its support for permuting and
transforming DOFs for higher-order elements. As described in
@dof-transformations, these operations are necessary when solving
problems on arbitrary meshes, as differences in how neighbouring cells
orient their sub-entities can otherwise cause issues.

# References
