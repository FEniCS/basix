import typing
from ._basixcpp import create_element as _cpp_create_element
from ._basixcpp import ElementFamily, CellType, LatticeType


def get_family(family: str, cell: str):
    """Get a Basix ElementFamily enum representing the family type on the given cell."""
    # Family names that are valid for all cells
    families = {
        "Lagrange": ElementFamily.P,
        "P": ElementFamily.P,
        "Discontinuous Lagrange": ElementFamily.DP,
        "DP": ElementFamily.DP,
        "Bubble": ElementFamily.Bubble,
    }
    # Family names that are valid on non-interval cells
    if cell != "interval":
        families.update({
            "RT": ElementFamily.RT,
            "Raviart-Thomas": ElementFamily.RT,
            "N1F": ElementFamily.RT,
            "N1div": ElementFamily.RT,
            "Nedelec 1st kind H(div)": ElementFamily.RT,
            "N1E": ElementFamily.N1E,
            "N1curl": ElementFamily.N1E,
            "Nedelec 1st kind H(curl)": ElementFamily.N1E,
            "BDM": ElementFamily.BDM,
            "Brezzi-Douglas-Marini": ElementFamily.BDM,
            "N2F": ElementFamily.BDM,
            "N2div": ElementFamily.BDM,
            "Nedelec 2nd kind H(div)": ElementFamily.BDM,
            "N2E": ElementFamily.N2E,
            "N2curl": ElementFamily.N2E,
            "Nedelec 2nd kind H(curl)": ElementFamily.N2E,
        })
    # Family names that are valid for tensor product cells
    if cell in ["interval", "quadrilateral", "hexahedron"]:
        families.update({
            "Q": ElementFamily.P,
            "DQ": ElementFamily.DP,
            "DPC": ElementFamily.DPC,
            "Serendipity": ElementFamily.Serendipity,
            "S": ElementFamily.Serendipity,
        })
    # Family names that are valid for quads and hexes
    if cell in ["quadrilateral", "hexahedron"]:
        families.update({
            "RTCF": ElementFamily.RT,
            "NCF": ElementFamily.RT,
            "RTCE": ElementFamily.N1E,
            "NCE": ElementFamily.N1E,
        })
    # Family names that are valid for triangles and tetrahedra
    if cell in ["triangle", "tetrahedron"]:
        families.update({
            "Regge": ElementFamily.Regge,
            "CR": ElementFamily.CR,
            "Crouzeix-Raviart": ElementFamily.CR,
        })

    if family in families:
        return families[family]

    raise ValueError(f"Unknown element family: {family} with cell type {cell}")


def string_to_cell(cell: str):
    """Convert a string to a Basix CellType enum."""
    if not hasattr(CellType, cell):
        raise ValueError(f"Unknown cell: {cell}")
    return getattr(CellType, cell)


def string_to_lattice(lattice: str):
    """Convert a string to a Basix LatticeType enum."""
    if not hasattr(LatticeType, lattice):
        raise ValueError(f"Unknown lattice type: {lattice}")
    return getattr(LatticeType, lattice)


def create_element(family : str, cell : str, degree : int,
                   lattice_type : typing.Optional[str]=None):
    """Create a finite element."""
    celltype = string_to_cell(cell)
    familytype = get_family(family, cell)

    if lattice_type is not None:
        return _cpp_create_element(familytype, celltype, degree, string_to_lattice(lattice_type))

    return _cpp_create_element(familytype, celltype, degree)
