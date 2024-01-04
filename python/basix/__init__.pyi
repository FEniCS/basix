from basix import cell as cell, finite_element as finite_element, lattice as lattice, polynomials as polynomials, quadrature as quadrature, sobolev_spaces as sobolev_spaces, variants as variants
from basix._basixcpp import CellType as CellType, DPCVariant as DPCVariant, ElementFamily as ElementFamily, LagrangeVariant as LagrangeVariant, LatticeSimplexMethod as LatticeSimplexMethod, LatticeType as LatticeType, MapType as MapType, PolynomialType as PolynomialType, PolysetType as PolysetType, QuadratureType as QuadratureType, SobolevSpace as SobolevSpace, __version__ as __version__, create_lattice as create_lattice, geometry as geometry, index as index, tabulate_polynomials as tabulate_polynomials, topology as topology
from basix.finite_element import create_custom_element as create_custom_element, create_element as create_element
from basix.quadrature import make_quadrature as make_quadrature

def compute_interpolation_operator(e0, e1): ...
