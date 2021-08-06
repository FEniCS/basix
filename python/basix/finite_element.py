"""Functions for creating finite elements."""

from ._basixcpp import ElementFamily, CellType, LatticeType


def string_to_family(family: str, cell: str):
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
