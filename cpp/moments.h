// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <Eigen/Dense>

namespace libtab
{

class FiniteElement;

/// ## Integral moments
/// These functions generate dual set matrices for integral moments
/// against spaces on a subentity of the cell
namespace moments
{
/// Make simple integral moments
///
/// This will compute the integral of each function in the moment space
/// over each sub entity of the moment space's cell type in a cell with
/// the given type. For example, if the input cell type is a triangle,
/// and the moment space is a P1 space on an edge, this will perform two
/// integrals for each of the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param poly_deg The polynomial degree of the poly set that defines the space
/// @param q_deg The quadrature degree used for the integrals
Eigen::MatrixXd make_integral_moments(const FiniteElement& moment_space,
                                      const cell::type celltype,
                                      const int value_size, const int poly_deg,
                                      const int q_deg);
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd> make_integral_moments_interpolation(
    const FiniteElement& moment_space, const cell::type celltype,
    const int value_size, const int poly_deg, const int q_deg);

/// Make dot product integral moments
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
Eigen::MatrixXd make_dot_integral_moments(const FiniteElement& moment_space,
                                          const cell::type celltype,
                                          const int value_size,
                                          const int poly_deg, const int q_deg);
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
make_dot_integral_moments_interpolation(const FiniteElement& moment_space,
                                        const cell::type celltype,
                                        const int value_size,
                                        const int poly_deg, const int q_deg);

/// Make tangential integral moments
///
/// These can only be used when the moment space is defined on edges of
/// the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// @param poly_deg The polynomial degree of the poly set that defines
/// the space
/// @param q_deg The quadrature degree used for the integrals
Eigen::MatrixXd make_tangent_integral_moments(const FiniteElement& moment_space,
                                              const cell::type celltype,
                                              const int value_size,
                                              const int poly_deg,
                                              const int q_deg);

/// Make interpolation points and weights for tangent integral moments
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
make_tangent_integral_moments_interpolation(const FiniteElement& moment_space,
                                            const cell::type celltype,
                                            const int value_size,
                                            const int poly_deg,
                                            const int q_deg);

/// Make normal integral moments
///
/// These can only be used when the moment space is defined on facets of
/// the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// @param poly_deg The polynomial degree of the poly set that defines
/// the space
/// @param q_deg The quadrature degree used for the integrals
// TODO: Implement this one in integral-moments.cpp
Eigen::MatrixXd make_normal_integral_moments(const FiniteElement& moment_space,
                                             const cell::type celltype,
                                             const int value_size,
                                             const int poly_deg,
                                             const int q_deg);

}; // namespace moments
} // namespace libtab
