
#include <Eigen/Core>

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

const char* mapping_name(int handle);

const std::vector<std::vector<int>>& entity_dofs(int handle);

const std::vector<Eigen::MatrixXd>& base_permutations(int handle);

const Eigen::ArrayXXd& points(int handle);

const Eigen::MatrixXd& interpolation_matrix(int handle);

/// Cell geometry
Eigen::ArrayXXd geometry(const char* cell_type);

/// Cell topology
std::vector<std::vector<std::vector<int>>> topology(const char* cell_type);

} // namespace basix
