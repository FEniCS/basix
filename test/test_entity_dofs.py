# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
from .utils import parametrize_over_elements


@parametrize_over_elements(4)
def test_number_of_dofs(cell_type, degree, element_type, element_args):
    element = basix.create_element(element_type, cell_type, degree, *element_args)

    entity_dofs = element.entity_dofs
    num_entity_dofs = element.num_entity_dofs

    for entity_dofs, num_entity_dofs in zip(element.entity_dofs, element.num_entity_dofs):
        for dofs, ndofs in zip(entity_dofs, num_entity_dofs):
            assert len(dofs) == ndofs

    assert element.dim == sum(sum(i) for i in element.num_entity_dofs)


@parametrize_over_elements(4)
def test_ordering_of_dofs(cell_type, degree, element_type, element_args):
    """
    Check that DOFs are numbered in ascending order by entity.

    This assumption is required for DOF transformations to work correctly.
    """
    element = basix.create_element(element_type, cell_type, degree, *element_args)

    entity_dofs = element.entity_dofs
    dof = 0
    for entity_dofs in element.entity_dofs:
        for dofs in entity_dofs:
            for d in sorted(dofs):
                assert d == dof
                dof += 1


@parametrize_over_elements(4)
def test_closure_dofs(cell_type, degree, element_type, element_args):
    element = basix.create_element(element_type, cell_type, degree, *element_args)

    entity_dofs = element.entity_dofs
    entity_closure_dofs = element.entity_closure_dofs

    topology = basix.topology(cell_type)
    connectivity = basix.cell.sub_entity_connectivity(cell_type)

    for dim, t in enumerate(topology):
        for e, _ in enumerate(t):
            for dim2 in range(dim + 1):
                for e2 in connectivity[dim][e][dim2]:
                    for dof in entity_dofs[dim2][e2]:
                        assert dof in entity_closure_dofs[dim][e]
