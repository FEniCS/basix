// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>

namespace libtab
{

namespace dofperms
{

Eigen::Array<int, Eigen::Dynamic, 1> interval_reflection(int degree);

Eigen::Array<int, Eigen::Dynamic, 1> triangle_reflection(int degree);

Eigen::Array<int, Eigen::Dynamic, 1> triangle_rotation(int degree);

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    interval_reflection_tangent_directions(int degree);

}; // namespace dofperms
} // namespace libtab
