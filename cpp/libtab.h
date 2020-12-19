
#include <string>
#include <Eigen/Core>

namespace libtab
{

/// Return handle
int register_element(std::string family_name, std::string cell_type, int degree);

/// Delete from global registry
void deregister_element(int handle);

/// Tabulate
std::vector<Eigen::ArrayXXd> tabulate(int handle, int nd, const Eigen::ArrayXXd& x);

}
