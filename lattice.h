#include <Eigen/Dense>

// Create a set of points on a line/triangle/tetrahedron in a grid
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_lattice(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>& vertices,
               int n);
