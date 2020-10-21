// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

namespace libtab
{

/// ## Integral moments
/// These functions generate dual set matrices for integral moments
/// against spaces on a subentity of the cell
namespace IntegralMoments
{
/// Make simple or dot product integral moments
///
/// This will compute the integral of each function in the moment space over
/// each sub entity of the moment space's cell type in a cell with the given
/// type. For example, if the input cell type is a triangle, and the moment
/// space is a P1 space on an edge, this will perform two integrals for each of
/// the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param poly_deg The polynomial degree of the poly set that defines the space
/// @param q_deg The quadrature degree used for the integrals
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
make_integral_moments(const FiniteElement& moment_space,
                      const Cell::Type celltype, const int value_size,
                      const int poly_deg, const int q_deg);

/// Make tangential integral moments
///
/// These can only be used when the moment space is defined on edges
/// of the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param poly_deg The polynomial degree of the poly set that defines the space
/// @param q_deg The quadrature degree used for the integrals
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
make_tangent_integral_moments(const FiniteElement& moment_space,
                              const Cell::Type celltype, const int value_size,
                              const int poly_deg, const int q_deg);

/// Make normal integral moments
///
/// These can only be used when the moment space is defined on facets
/// of the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param poly_deg The polynomial degree of the poly set that defines the space
/// @param q_deg The quadrature degree used for the integrals
// TODO: Implement this one in integral-moments.cpp
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
make_normal_integral_moments(const FiniteElement& moment_space,
                             const Cell::Type celltype, const int value_size,
                             const int poly_deg, const int q_deg);

}; // namespace IntegralMoments
} // namespace libtab
