====================================
Basix Python interface documentation
====================================

Welcome to the Basix Python interface documentation.

Basix is a finite element definition and tabulation runtime library. It
is part of `FEniCSx <https://docs.fenicsproject.org>`_, alongside `UFL
<https://fenics.readthedocs.io/projects/ufl/en/latest>`_, `FFCx
<https://docs.fenicsproject.org/ffcx/main>`_ and DOLFINx (`C++ docs
<https://docs.fenicsproject.org/dolfinx/main/cpp>`_, `Python docs
<https://docs.fenicsproject.org/dolfinx/main/python>`_).


Basix can create finite elements on intervals, triangles,
quadrilaterals, tetrahedra, hexahedra, prisms, and pyramids.

Using Basix
===========
A Basix element can be created using the function
:meth:`basix.create_element`. This function will return a
:class:`basix.finite_element.FiniteElement` object.

The element can be tabulated using the function
:meth:`basix.finite_element.FiniteElement.tabulate`.


Table of contents
=================
.. autosummary::
   :toctree: _autosummary
   :recursive:

   basix
