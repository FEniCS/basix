"""Functions to interpolate between elements."""
import numpy


def compute_interpolation_between_elements(element1, element2):
    assert element1.cell_type == element2.cell_type
    assert element1.value_size == element2.value_size

    points = element2.points
    tab = element1.tabulate(0, points)[0]

    # TODO: value size
    tab = numpy.vstack([tab[:, i * element1.dim: (i + 1) * element1.dim]
                        for i in range(element1.value_size)])

    return element2.interpolation_matrix @ tab
