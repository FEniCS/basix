from basix._basixcpp import CellType as _CT

__all__ = ['string_to_type', 'type_to_string']

def string_to_type(cell: str) -> _CT: ...
def type_to_string(celltype: _CT) -> str: ...