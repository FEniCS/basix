"""Functions to interpolate between elements."""
import numpy


def compute_interpolation_between_elements(element1, element2):
    """Compute an matrix to interpolate a function from element1 to element2."""
    assert element1.cell_type == element2.cell_type

    if element1.value_size != element2.value_size:
        if element2.value_size == 1:
            # Map element1's components into element2
            points = element2.points
            tab = element1.tabulate(0, points)[0]

            out = numpy.empty((element2.dim * element1.value_size, element1.dim))
            for i in range(element1.value_size):
                out[i::element1.value_size, :] = element2.interpolation_matrix @ tab[
                    :, i * element1.dim: (i + 1) * element1.dim]
            return out
        elif element1.value_size == 1:
            # Map duplicates of element2 to components of element2
            points = element2.points
            tab = element1.tabulate(0, points)[0]

            out = numpy.zeros((element2.dim, element1.dim * element2.value_size))
            for i in range(element2.value_size):
                out[:, i::element2.value_size] += element2.interpolation_matrix[
                    :, i * tab.shape[0]:(i + 1) * tab.shape[0]] @ tab
            return out
        else:
            raise RuntimeError("Cannot interpolation between elements with this combination of value sizes.")

    assert element1.value_size == element2.value_size

    points = element2.points
    tab = element1.tabulate(0, points)[0]

    tab = numpy.vstack([tab[:, i * element1.dim: (i + 1) * element1.dim]
                        for i in range(element1.value_size)])

    return element2.interpolation_matrix @ tab
