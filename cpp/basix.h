// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Core>
#include <complex>
#include <utility>
#include <vector>

namespace basix
{

/// Create element in global registry and return handle
int register_element(const char* family_name, const char* cell_type,
                     int degree);

/// Delete from global registry
void release_element(int handle);

/// Tabulate
std::vector<Eigen::ArrayXXd> tabulate(int handle, int nd,
                                      const Eigen::ArrayXXd& x);

/// Map a function value from the reference to a physical cell
/// @param[in] handle The handle of the basix element
/// @param[in] reference_data The reference data at a single point
/// @param[in] J The Jacobian of the map to the cell (evaluated at the point)
/// @param[in] detJ The determinant of the Jacobian of the map to the cell
/// (evaluated at the point)
/// @param[in] K The inverse of the Jacobian of the map to the cell (evaluated
/// at the point)
/// @return The data on the physical cell at the corresponding point
Eigen::Array<double, Eigen::Dynamic, 1> map_push_forward_real(
    int handle, const Eigen::Array<double, Eigen::Dynamic, 1>& reference_data,
    const Eigen::MatrixXd& J, double detJ, const Eigen::MatrixXd& K);
Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> map_push_forward_complex(
    int handle,
    const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>& reference_data,
    const Eigen::MatrixXd& J, double detJ, const Eigen::MatrixXd& K);

/// Map a function value from a physical cell to the reference
/// @param[in] handle The handle of the basix element
/// @param[in] physical_data The physical data at a single point
/// @param[in] J The Jacobian of the map to the cell (evaluated at the point)
/// @param[in] detJ The determinant of the Jacobian of the map to the cell
/// (evaluated at the point)
/// @param[in] K The inverse of the Jacobian of the map to the cell (evaluated
/// at the point)
/// @return The data on the reference element at the corresponding point
Eigen::Array<double, Eigen::Dynamic, 1> map_pull_back_real(
    int handle, const Eigen::Array<double, Eigen::Dynamic, 1>& physical_data,
    const Eigen::MatrixXd& J, double detJ, const Eigen::MatrixXd& K);
Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> map_pull_back_complex(
    int handle,
    const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>& physical_data,
    const Eigen::MatrixXd& J, double detJ, const Eigen::MatrixXd& K);

/// Cell type
const char* cell_type(int handle);

/// Degree
int degree(int handle);

/// Value size
int value_size(int handle);

///  Value shape
const std::vector<int>& value_shape(int handle);

/// Finite Element dimension
int dim(int handle);

/// Family name
const char* family_name(int handle);

/// Mapping name (identity, piola etc.)
const char* mapping_name(int handle);

/// Number of dofs per entity, ordered from vertex, edge, facet, cell
const std::vector<std::vector<int>>& entity_dofs(int handle);

/// Base permutations
const std::vector<Eigen::MatrixXd>& base_permutations(int handle);

/// Interpolation points
const Eigen::ArrayXXd& points(int handle);

/// Interpolation matrix
const Eigen::MatrixXd& interpolation_matrix(int handle);

/// Cell geometry
Eigen::ArrayXXd geometry(const char* cell_type);

/// Cell topology
std::vector<std::vector<std::vector<int>>> topology(const char* cell_type);

/// Create quadrature points and weights
std::pair<Eigen::ArrayXXd, Eigen::ArrayXd>
make_quadrature(const char* rule, const char* cell_type, int order);

} // namespace basix
