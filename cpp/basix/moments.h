// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <Eigen/Dense>
#include <xtensor/xtensor.hpp>

namespace basix
{

class FiniteElement;

/// ## Integral moments
/// These functions generate dual set matrices for integral moments
/// against spaces on a subentity of the cell
namespace moments
{
/// Create the dof transformations for an integral moment.
///
/// If the moment space is an interval, this returns one matrix representing the
/// reversal of the interval. If the moment space is a face, this returns two
/// matrices: one representing a rotation, the other a reflection
///
/// @param[in] moment_space The finite element space that the integral moment is
/// taken against
/// @return A list of dof transformations
std::vector<Eigen::MatrixXd>
create_moment_dof_transformations(const FiniteElement& moment_space);

/// Create the dof transformations for a dot integral moment.
///
/// @param[in] moment_space The finite element space that the integral moment is
/// taken against
/// @return A list of dof transformations
std::vector<Eigen::MatrixXd>
create_dot_moment_dof_transformations(const FiniteElement& moment_space);

/// Create the dof transformations for an integral moment.
///
/// If the moment space is an interval, this returns one matrix representing the
/// reversal of the interval. If the moment space is a face, this returns two
/// matrices: one representing a rotation, the other a reflection
///
/// @param[in] moment_space The finite element space that the integral moment is
/// taken against
/// @return A list of dof transformations
xt::xtensor<double, 3>
create_dot_moment_dof_transformations_new(const FiniteElement& moment_space);

/// Create the dof transformations for a normal integral moment.
///
/// @param[in] moment_space The finite element space that the integral moment is
/// taken against
/// @return A list of dof transformations
std::vector<Eigen::MatrixXd>
create_normal_moment_dof_transformations(const FiniteElement& moment_space);

/// Create the dof transformations for a tangential integral moment.
///
/// @param[in] moment_space The finite element space that the integral moment is
/// taken against
/// @return A list of dof transformations
std::vector<Eigen::MatrixXd>
create_tangent_moment_dof_transformations(const FiniteElement& moment_space);

/// Make interpolation points and weights for simple integral moments
///
/// These will represent the integral of each function in the moment space
/// over each sub entity of the moment space's cell type in a cell with
/// the given type. For example, if the input cell type is a triangle,
/// and the moment space is a P1 space on an edge, this will perform two
/// integrals for each of the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
make_integral_moments(const FiniteElement& moment_space, cell::type celltype,
                      std::size_t value_size, int q_deg);

/// Make interpolation points and weights for simple integral moments
///
/// These will represent the integral of each function in the moment space
/// over each sub entity of the moment space's cell type in a cell with
/// the given type. For example, if the input cell type is a triangle,
/// and the moment space is a P1 space on an edge, this will perform two
/// integrals for each of the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_integral_moments_new(const FiniteElement& moment_space,
                          cell::type celltype, std::size_t value_size,
                          int q_deg);

/// Make interpolation points and weights for simple integral moments
///
/// These will represent the integral of each function in the moment space
/// over each sub entity of the moment space's cell type in a cell with
/// the given type. For example, if the input cell type is a triangle,
/// and the moment space is a P1 space on an edge, this will perform two
/// integrals for each of the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_integral_moments_new(const FiniteElement& moment_space,
                          cell::type celltype, std::size_t value_size,
                          int q_deg);

/// Make interpolation points and weights for dot product integral moments
///
/// These will represent the integral of each function in the moment space over
/// each sub entity of the moment space's cell type in a cell with the given
/// type. For example, if the input cell type is a triangle, and the moment
/// space is a P1 space on an edge, this will perform two integrals for each of
/// the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
make_dot_integral_moments(const FiniteElement& moment_space,
                          cell::type celltype, int value_size, int q_deg);

/// Make interpolation points and weights for dot product integral moments
///
/// These will represent the integral of each function in the moment space over
/// each sub entity of the moment space's cell type in a cell with the given
/// type. For example, if the input cell type is a triangle, and the moment
/// space is a P1 space on an edge, this will perform two integrals for each of
/// the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_dot_integral_moments_new(const FiniteElement& moment_space,
                              cell::type celltype, std::size_t value_size,
                              int q_deg);

/// Make interpolation points and weights for tangent integral moments
///
/// These can only be used when the moment space is defined on edges of
/// the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// the space
/// @param q_deg The quadrature degree used for the integrals
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
make_tangent_integral_moments(const FiniteElement& moment_space,
                              cell::type celltype, int value_size, int q_deg);

/// Make interpolation points and weights for tangent integral moments
///
/// These can only be used when the moment space is defined on edges of
/// the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// the space
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_tangent_integral_moments_new(const FiniteElement& moment_space,
                                  cell::type celltype, std::size_t value_size,
                                  int q_deg);

/// Make interpolation points and weights for normal integral moments
///
/// These can only be used when the moment space is defined on facets of
/// the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// the space
/// @param q_deg The quadrature degree used for the integrals
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
make_normal_integral_moments(const FiniteElement& moment_space,
                             cell::type celltype, int value_size, int q_deg);

/// Make interpolation points and weights for normal integral moments
///
/// These can only be used when the moment space is defined on facets of
/// the cell
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// the space
/// @param q_deg The quadrature degree used for the integrals
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
make_normal_integral_moments_new(const FiniteElement& moment_space,
                                 cell::type celltype, std::size_t value_size,
                                 int q_deg);
} // namespace moments
} // namespace basix
