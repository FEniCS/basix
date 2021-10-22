====================================
Basix Python interface documentation
====================================

Welcome to the Basix Python interface documentation.

Basix is a finite element definition and tabulation runtime library.
It is part of `FEniCSx <https://docs.fenicsproject.org>`_,
alongside `FFCx <https://docs.fenicsproject.org/ffcx>`_ and `DOLFINx <https://docs.fenicsproject.org/dolfinx/cpp>`_.

Basix can create finite elements on intervals, triangles, quadrilaterals, tetrahedra, hexahedra, prisms, and pyramids.

Using Basix
===========
A Basix element can be created using the function :meth:`basix.create_element`.
This function will return a :class:`basix.finite_element.FiniteElement` object.

The element can be tabulated using the function :meth:`basix.finite_element.FiniteElement.tabulate`.


Table of contents
=================
.. autosummary::
   :toctree: _autosummary
   :recursive:

   basix
