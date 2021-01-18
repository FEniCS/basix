
#include <Eigen/Core>
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
void tabulate(int handle, double* basis_values, int nd, const double* x,
              int npoints, int tdim);

/// Cell type
const char* cell_type(int handle);

/// Degree
int degree(int handle);

/// Value rank
/// @param handle Identifier
/// @return The number of dimensions of the value shape
int value_rank(int handle);

/// Value shape
/// @param handle Identifier
/// @param[in/out] dimensions Array of value_rank size
void value_shape(int handle, int* dimensions);

/// Finite Element dimension
int dim(int handle);

/// Family name
const char* family_name(int handle);

/// Mapping name (affine, piola etc.)
const char* mapping_name(int handle);

/// Number of dofs per entity of given dimension
/// @param handle Identifier
/// @param dim Entity dimension
/// @param[in/out] num_dofs Number of dofs on each entity
void entity_dofs(int handle, int dim, int* num_dofs);

/// Base permutations
const std::vector<Eigen::MatrixXd>& base_permutations(int handle);

/// Interpolation points
const Eigen::ArrayXXd& points(int handle);

/// Interpolation matrix
const Eigen::MatrixXd& interpolation_matrix(int handle);

/// Cell geometry
int cell_geometry_num_points(const char* cell_type);
int cell_geometry_dimension(const char* cell_type);
void geometry(const char* cell_type, double* points);

/// Cell topology
std::vector<std::vector<std::vector<int>>> topology(const char* cell_type);

/// Create quadrature points and weights
std::pair<Eigen::ArrayXXd, Eigen::ArrayXd>
make_quadrature(const char* rule, const char* cell_type, int order);

} // namespace basix
