try:
    import numba
    import numpy
except ImportError:
    raise RuntimeError("You must have numba installed to use the numba helper functions.")


@numba.njit
def apply_dof_transformation(tdim, edge_count, face_count, entity_transformations, entity_dofs,
                             data, block_size, cell_info):
    """Apply dof transformations to some data.

    Parameters
    ----------
    tdim : int
        The topological dimension of the cell.
    edge_cout : int
        The number of edges the cell has.
    face_count : int
        The number of faces the cell has.
    entity_transformations : list
        The DOF transformations for each entity.
    entity_dofs : list
        The number of DOFs on each entity.
    data : np.array
        The data. This will be changed by this function.
    block_size : int
        The number of data entries for each DOF.
    cell_info : int
        An integer representing the orientations of the subentities of the cell.
    """
    if tdim >= 2:
        if tdim == 3:
            face_start = 3 * face_count
        else:
            face_start = 0

        dofstart = 0
        for i in entity_dofs[0]:
            dofstart += i
        # NOTE: Copy array to make numba compilation faster (contiguous array assumption)
        edge_reflection = entity_transformations[0].copy()
        for e in range(edge_count):
            edofs = entity_dofs[1][e]
            if edofs == 0:
                continue
            if cell_info >> (face_start + e) & 1:
                data[dofstart:dofstart+edofs] = numpy.dot(edge_reflection, data[dofstart:dofstart+edofs])
            dofstart += edofs

        if tdim == 3:
            face_rotation = entity_transformations[1].copy()
            face_reflection = entity_transformations[2].copy()
            for f in range(face_count):
                fdofs = entity_dofs[2][f]
                if fdofs == 0:
                    continue
                if cell_info >> (3 * f) & 1:
                    data[dofstart:dofstart+fdofs] = numpy.dot(face_reflection, data[dofstart:dofstart+fdofs])
                for _ in range(cell_info >> (3 * f + 1) & 3):
                    data[dofstart:dofstart+fdofs] = numpy.dot(face_rotation, data[dofstart:dofstart+fdofs])
                dofstart += fdofs


@numba.njit
def apply_dof_transformation_interval(entity_transformations, entity_dofs,
                                      data, block_size, cell_info):
    """Apply dof transformations to some data on an interval.

    Parameters
    ----------
    entity_transformations : Dict(ndarray(float64))
        The DOF transformations for each entity.
    entity_dofs : Dict(ndarray(int32))
        The number of DOFs on each entity.
    data : np.array
        The data. This will be changed by this function.
    block_size : int
        The number of data entries for each DOF.
    cell_info : int
        An integer representing the orientations of the subentities of the cell.
    """
    return


@numba.njit
def apply_dof_transformation_triangle(entity_transformations, entity_dofs,
                                      data, block_size, cell_info):
    """Apply dof transformations to some data on a triangle.

    Parameters
    ----------
    entity_transformations : Dict(ndarray(float64))
        The DOF transformations for each entity.
    entity_dofs : Dict(ndarray(int32))
        The number of DOFs on each entity.
    data : np.array
        The data. This will be changed by this function.
    block_size : int
        The number of data entries for each DOF.
    cell_info : int
        An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(2, 3, 1, entity_transformations, entity_dofs,
                             data, block_size, cell_info)


@numba.njit
def apply_dof_transformation_quadrilateral(
    entity_transformations, entity_dofs, data, block_size, cell_info
):
    """Apply dof transformations to some data on an quadrilateral.

    Parameters
    ----------
    entity_transformations : Dict(ndarray(float64))
        The DOF transformations for each entity.
    entity_dofs : Dict(ndarray(int32))
        The number of DOFs on each entity.
    data : np.array
        The data. This will be changed by this function.
    block_size : int
        The number of data entries for each DOF.
    cell_info : int
        An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(2, 4, 1, entity_transformations, entity_dofs,
                             data, block_size, cell_info)


@numba.njit
def apply_dof_transformation_tetrahedron(
    entity_transformations, entity_dofs, data, block_size, cell_info
):
    """Apply dof transformations to some data on a tetrahedron.

    Parameters
    ----------
    entity_transformations : Dict(ndarray(float64))
        The DOF transformations for each entity.
    entity_dofs : Dict(ndarray(int32))
        The number of DOFs on each entity.
    data : np.array
        The data. This will be changed by this function.
    block_size : int
        The number of data entries for each DOF.
    cell_info : int
        An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(3, 6, 4, entity_transformations, entity_dofs,
                             data, block_size, cell_info)


@numba.njit
def apply_dof_transformation_hexahedron(
    entity_transformations, entity_dofs, data, block_size, cell_info
):
    """Apply dof transformations to some data on an hexahedron.

    Parameters
    ----------
    entity_transformations : Dict(ndarray(float64))
        The DOF transformations for each entity.
    entity_dofs : Dict(ndarray(int32))
        The number of DOFs on each entity.
    data : np.array
        The data. This will be changed by this function.
    block_size : int
        The number of data entries for each DOF.
    cell_info : int
        An integer representing the orientations of the subentities of the cell.
    """
    apply_dof_transformation(3, 12, 6, entity_transformations, entity_dofs,
                             data, block_size, cell_info)
