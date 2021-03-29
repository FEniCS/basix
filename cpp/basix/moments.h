// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <xtensor/xtensor.hpp>

namespace basix
{

class FiniteElement;

/// ## Integral moments
/// These functions generate dual set matrices for integral moments
/// against spaces on a subentity of the cell
namespace moments
{

/// Create the dof transformations for the DOFs defined using a dot integral
/// moment.
///
/// A dot integral moment is defined by
/// \f[l_i(v) = \int v\cdot\phi_i,\f]
/// where \f$\phi_i\f$ is a basis function in the moment space, and \f$v\f$ and
/// \f$\phi_i\f$ are either both scalars or are vectors of the same size.
///
/// If the moment space is an interval, this returns one matrix
/// representing the reversal of the interval. If the moment space is a
/// face, this returns two matrices: one representing a rotation, the
/// other a reflection.
///
/// These matrices are computed by calculation the interpolation coefficients
/// of a rotated/reflected basis into the original basis.
///
/// @param[in] moment_space The finite element space that the integral
/// moment is taken against
/// @return A list of dof transformations
xt::xtensor<double, 3>
create_dot_moment_dof_transformations(const FiniteElement& moment_space);

/// Create the DOF transformations for the DOFs defined using an integral
/// moment.
///
/// An integral moment is defined by
/// \f[l_{i,j}(v) = \int v\cdot e_j\phi_i,\f]
/// where \f$\phi_i\f$ is a basis function in the moment space, \f$e_j\f$ is a
/// coordinate direction (of the cell sub-entity the moment is taken on),
/// \f$v\f$ is a vector, and \f$\phi_i\f$ is a scalar.
///
/// This will combine multiple copies of the result of
/// `create_dot_moment_dof_transformations` to give the transformations for
/// integral moments of each vector component against the moment space.
///
/// @param[in] moment_space The finite element space that the integral
/// moment is taken against
/// @return A list of dof transformations
xt::xtensor<double, 3>
create_moment_dof_transformations(const FiniteElement& moment_space);

/// Create the dof transformations for the DOFs defined using a normal integral
/// moment.
///
/// A normal integral moment is defined by
/// \f[l_{i,j}(v) = \int v\cdot n\phi_i,\f]
/// where \f$\phi_i\f$ is a basis function in the moment space, \f$n\f$ is
/// normal to the cell sub-entity, \f$v\f$ is a vector, and \f$\phi_i\f$ is a
/// scalar.
///
/// This does the same as `create_dot_moment_dof_transformations` with some
/// additional factors of -1 to account for the changing of the normal direction
/// when the entity is reflected.
///
/// @param[in] moment_space The finite element space that the integral
/// moment is taken against
/// @return A list of dof transformations
xt::xtensor<double, 3>
create_normal_moment_dof_transformations(const FiniteElement& moment_space);

/// Create the dof transformations for the DOFs defined using a tangential
/// integral moment.
///
/// A tangential integral moment is defined by
/// \f[l_{i,j}(v) = \int v\cdot t\phi_i,\f]
/// where \f$\phi_i\f$ is a basis function in the moment space, \f$t\f$ is
/// tangential to the edge, \f$v\f$ is a vector, and \f$\phi_i\f$ is a scalar.
///
/// This does the same as `create_dot_moment_dof_transformations` with some
/// additional factors of -1 to account for the changing of the tangent
/// direction when the edge is reflected.
///
/// @param[in] moment_space The finite element space that the integral
/// moment is taken against
/// @return A list of dof transformations
xt::xtensor<double, 3>
create_tangent_moment_dof_transformations(const FiniteElement& moment_space);

/// Make interpolation points and weights for simple integral moments
///
/// These will represent the integral of each function in the moment
/// space over each sub entity of the moment space's cell type in a cell
/// with the given type. For example, if the input cell type is a
/// triangle, and the moment space is a P1 space on an edge, this will
/// perform two integrals for each of the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_integral_moments(const FiniteElement& moment_space, cell::type celltype,
                      std::size_t value_size, int q_deg);

/// TODO
std::pair<xt::xtensor<double, 3>, xt::xtensor<double, 4>>
make_integral_moments_new(const FiniteElement& moment_space,
                          cell::type celltype, std::size_t value_size,
                          int q_deg);

/// Make interpolation points and weights for dot product integral
/// moments
///
/// These will represent the integral of each function in the moment
/// space over each sub entity of the moment space's cell type in a cell
/// with the given type. For example, if the input cell type is a
/// triangle and the moment space is a P1 space on an edge, this will
/// perform two integrals for each of the 3 edges of the triangle.
///
/// @todo Clarify what happens value size of the moment space is less
/// than `value_size`.
///
/// @param V The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_dot_integral_moments(const FiniteElement& V, cell::type celltype,
                          std::size_t value_size, int q_deg);

/// TODO
std::pair<xt::xtensor<double, 3>, xt::xtensor<double, 4>>
make_dot_integral_moments_new(const FiniteElement& V, cell::type celltype,
                              std::size_t value_size, int q_deg);

/// Make interpolation points and weights for tangent integral moments
///
/// These can only be used when the moment space is defined on edges of
/// the cell
///
/// @param V The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined the
/// space
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_tangent_integral_moments(const FiniteElement& V, cell::type celltype,
                              std::size_t value_size, int q_deg);

/// Compute interpolation points and weights for normal integral moments
///
/// These can only be used when the moment space is defined on facets of
/// the cell
///
/// @param[in] V The space to compute the integral moments against
/// @param[in] celltype The cell type of the cell on which the space is
/// being defined
/// @param[in] value_size The value size of the space being defined
/// @param[in] q_deg The quadrature degree used for the integrals
/// @return (interpolation points, interpolation matrix)
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_normal_integral_moments(const FiniteElement& V, cell::type celltype,
                             std::size_t value_size, int q_deg);

/// TODO
std::pair<xt::xtensor<double, 3>, xt::xtensor<double, 4>>
make_normal_integral_moments_new(const FiniteElement& V, cell::type celltype,
                                 std::size_t value_size, int q_deg);

} // namespace moments
} // namespace basix
