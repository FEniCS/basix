# Copyright (c) 2021 Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
from .utils import parametrize_over_elements


@parametrize_over_elements(4)
def test_number_of_dofs(cell_name, order, element_name):
    element = basix.create_element(element_name, cell_name, order)

    entity_dofs = element.entity_dofs
    entity_dof_counts = element.entity_dof_counts

    for entity_dofs, entity_dof_counts in zip(element.entity_dofs, element.entity_dof_counts):
        for dofs, ndofs in zip(entity_dofs, entity_dof_counts):
            assert len(dofs) == ndofs


@parametrize_over_elements(4)
def test_ordering_of_dofs(cell_name, order, element_name):
    """
    Check that DOFs are numbered in ascending order by entity.

    This assumption is required for DOF transformations to work correctly.
    """
    element = basix.create_element(element_name, cell_name, order)

    entity_dofs = element.entity_dofs
    dof = 0

    for entity_dofs in element.entity_dofs:
        for dofs in entity_dofs:
            for d in dofs:
                assert d == dof
                dof += 1
