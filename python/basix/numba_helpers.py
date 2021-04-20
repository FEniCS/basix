try:
    import numba
except ImportError:
    raise RuntimeError("You must have numba installed to use the numba helper functions.")


@numba.njit
def apply_dof_transformation(entity_transformations, cell_type, entity_dofs,
                             data, block_size, cell_info):
    if cell_type == "interval":
        tdim = 1
    elif cell_type == "triangle":
        edge_count = 3
        tdim = 2
    elif cell_type == "quadrilateral":
        edge_count = 4
        tdim = 2
    elif cell_type == "tetrahedron":
        edge_count = 6
        face_count = 4
        tdim = 3
    elif cell_type == "hexahedron":
        edge_count = 12
        face_count = 6
        tdim = 3

    if tdim >= 2:
        if tdim == 3:
            face_start = 3 * face_count
        else:
            face_start = 0

        dofstart = 0
        for i in entity_dofs[0]:
            dofstart += i

        for e in range(edge_count):
            edofs = entity_dofs[1][e]
            if cell_info >> (face_start + e) & 1:
                for b in range(block_size):
                    s = (dofstart * block_size + b, (dofstart + edofs) * block_size, block_size)
                    data[slice(*s)] = entity_transformations[0].dot(data[slice(*s)])
            dofstart += edofs

        if tdim == 3:
            for f in range(face_count):
                fdofs = entity_dofs[2][f]
                if cell_info >> (3 * f) & 1:
                    for b in range(block_size):
                        s = (dofstart * block_size + b, (dofstart + fdofs) * block_size, block_size)
                        data[slice(*s)] = entity_transformations[2].dot(data[slice(*s)])
                for _ in range(cell_info >> (3 * f + 1) & 3):
                    for b in range(block_size):
                        s = (dofstart * block_size + b, (dofstart + fdofs) * block_size, block_size)
                        data[slice(*s)] = entity_transformations[1].dot(data[slice(*s)])
                dofstart += fdofs
