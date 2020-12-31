
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
std::vector<Eigen::ArrayXXd> tabulate(int handle, int nd,
                                      const Eigen::ArrayXXd& x);

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

/// Mapping name (affine, piola etc.)
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
