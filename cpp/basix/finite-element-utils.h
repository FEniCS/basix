// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"
#include "types.h"
#include <array>
#include <vector>

namespace basix
{

namespace element
{
/// Typedef for mdspan
template <typename T, std::size_t d>
using mdspan_t = impl::mdspan_t<T, d>;

/// Create a version of the interpolation points, interpolation
/// matrices and entity transformation that represent a discontinuous
/// version of the element. This discontinuous version will have the
/// same DOFs but they will all be associated with the interior of the
/// reference cell.
/// @param[in] x Interpolation points. Indices are (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index, derivative)
/// @param[in] tdim The topological dimension of the cell the element is
/// defined on
/// @param[in] value_size The value size of the element
/// @return (xdata, xshape, Mdata, Mshape), where the x and M data are
/// for  a discontinuous version of the element (with the same shapes as
/// x and M)
template <std::floating_point T>
std::tuple<std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 2>>, 4>,
           std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 4>>, 4>>
make_discontinuous(const std::array<std::vector<mdspan_t<const T, 2>>, 4>& x,
                   const std::array<std::vector<mdspan_t<const T, 4>>, 4>& M,
                   std::size_t tdim, std::size_t value_size);

/// Compute the dual matrix for an element
/// @param[in] cell_type The cell type
/// @param[in] poly_type The polynomial set type
/// @param[in] B coefficients defining the polynomial space for this element
/// @param[in] x Interpolation points
/// @param[in] M Interpolation operators
/// @param[in] degree Polynomial degree
/// @param[in] nderivs Number of derivatives used in interpolation
/// @return the data and shape of the dual matrix
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>> compute_dual_matrix(
    cell::type cell_type, polyset::type poly_type, mdspan_t<const T, 2> B,
    const std::array<std::vector<impl::mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<impl::mdspan_t<const T, 4>>, 4>& M, int degree,
    int nderivs);

} // namespace element

/// Create a custom finite element
/// @param[in] cell_type The cell type
/// @param[in] value_shape The value shape of the element
/// @param[in] wcoeffs Matrices for the kth value index containing the
/// expansion coefficients defining a polynomial basis spanning the
/// polynomial space for this element. Shape is (dim(finite element polyset),
/// dim(Legendre polynomials))
/// @param[in] x Interpolation points. Indices are (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index, derivative)
/// @param[in] interpolation_nderivs The number of derivatives that need to be
/// used during interpolation
/// @param[in] map_type The type of map to be used to map values from
/// the reference to a cell
/// @param[in] sobolev_space The underlying Sobolev space for the element
/// @param[in] discontinuous Indicates whether or not this is the
/// discontinuous version of the element
/// @param[in] embedded_subdegree The highest degree n such that a
/// Lagrange (or vector Lagrange) element of degree n is a subspace of this
/// element
/// @param[in] embedded_superdegree The degree of a polynomial in this element's
/// polyset
/// @param[in] poly_type The type of polyset to use for this element
/// @return A custom finite element
template <std::floating_point T>
FiniteElement<T> create_custom_element(
    cell::type cell_type, const std::vector<std::size_t>& value_shape,
    impl::mdspan_t<const T, 2> wcoeffs,
    const std::array<std::vector<impl::mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<impl::mdspan_t<const T, 4>>, 4>& M,
    int interpolation_nderivs, maps::type map_type,
    sobolev::space sobolev_space, bool discontinuous, int embedded_subdegree,
    int embedded_superdegree, polyset::type poly_type);

/// Create an element using a given Lagrange variant and a given DPC variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @param[in] dof_ordering Ordering of dofs for ElementDofLayout
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_element(element::family family, cell::type cell,
                                int degree, element::lagrange_variant lvariant,
                                element::dpc_variant dvariant,
                                bool discontinuous,
                                std::vector<int> dof_ordering = {});

/// Get the tensor product DOF ordering for an element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on.
/// Currently limited to quadrilateral or hexahedron.
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A vector containing the dof ordering
std::vector<int> tp_dof_ordering(element::family family, cell::type cell,
                                 int degree, element::lagrange_variant lvariant,
                                 element::dpc_variant dvariant,
                                 bool discontinuous);

/// Get the tensor factors of an element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on.
/// Currently limited to quadrilateral or hexahedron.
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @param[in] dof_ordering The ordering of the DOFs
/// @return A list of lists of finite element factors
template <std::floating_point T>
std::vector<std::vector<FiniteElement<T>>>
tp_factors(element::family family, cell::type cell, int degree,
           element::lagrange_variant lvariant, element::dpc_variant dvariant,
           bool discontinuous, std::vector<int> dof_ordering);

/// Create an element with Tensor Product dof ordering
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on.
/// Currently limited to quadrilateral or hexahedron.
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A finite element
template <std::floating_point T>
FiniteElement<T>
create_tp_element(element::family family, cell::type cell, int degree,
                  element::lagrange_variant lvariant,
                  element::dpc_variant dvariant, bool discontinuous);

} // namespace basix
