from basix._basixcpp import LatticeSimplexMethod as _LSM, LatticeType as _LT

__all__ = ['string_to_type', 'type_to_string', 'string_to_simplex_method', 'simplex_method_to_string']

def string_to_type(lattice: str) -> _LT: ...
def type_to_string(latticetype: _LT) -> str: ...
def string_to_simplex_method(method: str) -> _LSM: ...
def simplex_method_to_string(simplex_method: _LSM) -> str: ...