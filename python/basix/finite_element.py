"""Functions for creating finite elements."""

from ._basixcpp import ElementFamily as _EF


def string_to_family(family: str, cell: str):
    """Get a Basix ElementFamily enum representing the family type on the given cell."""
    # Family names that are valid for all cells
    families = {
        "Lagrange": _EF.P,
        "P": _EF.P,
        "Bubble": _EF.Bubble,
    }
    # Family names that are valid on non-interval cells
    if cell != "interval":
        families.update({
            "RT": _EF.RT,
            "Raviart-Thomas": _EF.RT,
            "N1F": _EF.RT,
            "N1div": _EF.RT,
            "Nedelec 1st kind H(div)": _EF.RT,
            "N1E": _EF.N1E,
            "N1curl": _EF.N1E,
            "Nedelec 1st kind H(curl)": _EF.N1E,
            "BDM": _EF.BDM,
            "Brezzi-Douglas-Marini": _EF.BDM,
            "N2F": _EF.BDM,
            "N2div": _EF.BDM,
            "Nedelec 2nd kind H(div)": _EF.BDM,
            "N2E": _EF.N2E,
            "N2curl": _EF.N2E,
            "Nedelec 2nd kind H(curl)": _EF.N2E,
        })
    # Family names that are valid for tensor product cells
    if cell in ["interval", "quadrilateral", "hexahedron"]:
        families.update({
            "Q": _EF.P,
            "DQ": _EF.DP,
            "DPC": _EF.DPC,
            "Serendipity": _EF.Serendipity,
            "S": _EF.Serendipity,
        })
    # Family names that are valid for quads and hexes
    if cell in ["quadrilateral", "hexahedron"]:
        families.update({
            "RTCF": _EF.RT,
            "NCF": _EF.RT,
            "RTCE": _EF.N1E,
            "NCE": _EF.N1E,
        })
    # Family names that are valid for triangles and tetrahedra
    if cell in ["triangle", "tetrahedron"]:
        families.update({
            "Regge": _EF.Regge,
            "CR": _EF.CR,
            "Crouzeix-Raviart": _EF.CR,
        })

    if family in families:
        return families[family]

    raise ValueError(f"Unknown element family: {family} with cell type {cell}")
