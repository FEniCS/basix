"""Functions for creating finite elements."""

from ._basixcpp import ElementFamily as _EF
from ._basixcpp import FiniteElement  # noqa: F401


def string_to_family(family: str, cell: str) -> _EF:
    """
    Get a Basix ElementFamily enum representing the family type on the given cell.

    Parameters
    ----------
    family : str
        The element family as a string.
    cell : str
        The cell type as a string.

    Returns
    -------
    basix.ElementFamily
        The element family.
    """
    # Family names that are valid for all cells
    families = {
        "Lagrange": _EF.p,
        "P": _EF.p,
        "Bubble": _EF.bubble,
    }
    # Family names that are valid on non-interval cells
    if cell != "interval":
        families.update({
            "RT": _EF.rt,
            "Raviart-Thomas": _EF.rt,
            "N1F": _EF.rt,
            "N1div": _EF.rt,
            "Nedelec 1st kind H(div)": _EF.rt,
            "N1E": _EF.n1e,
            "N1curl": _EF.n1e,
            "Nedelec 1st kind H(curl)": _EF.n1e,
            "BDM": _EF.bdm,
            "Brezzi-Douglas-Marini": _EF.bdm,
            "N2F": _EF.bdm,
            "N2div": _EF.bdm,
            "Nedelec 2nd kind H(div)": _EF.bdm,
            "N2E": _EF.n2e,
            "N2curl": _EF.n2e,
            "Nedelec 2nd kind H(curl)": _EF.n2e,
        })
    # Family names that are valid for tensor product cells
    if cell in ["interval", "quadrilateral", "hexahedron"]:
        families.update({
            "Q": _EF.p,
            "DPC": _EF.DPC,
            "Serendipity": _EF.serendipity,
            "serendipity": _EF.serendipity,
            "S": _EF.serendipity,
        })
    # Family names that are valid for quads and hexes
    if cell in ["quadrilateral", "hexahedron"]:
        families.update({
            "RTCF": _EF.rt,
            "NCF": _EF.rt,
            "RTCE": _EF.n1e,
            "NCE": _EF.n1e,
            "BDMCF": _EF.bdm,
            "BDMCE": _EF.n2e,
        })
    # Family names that are valid for triangles and tetrahedra
    if cell in ["triangle", "tetrahedron"]:
        families.update({
            "Regge": _EF.regge,
            "CR": _EF.cr,
            "Crouzeix-Raviart": _EF.cr,
        })

    if family in families:
        return families[family]

    raise ValueError(f"Unknown element family: {family} with cell type {cell}")
