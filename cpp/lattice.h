#include <Eigen/Dense>
#include <vector>
#include "cell.h"

#pragma once

namespace libtab
{
namespace lattice
{
/// Create a lattice of points on the cell, equispaced along each edge
/// optionally including the outer surface points
/// @param celltype The cell::Type
/// @param n number of points in each direction
/// @param exterior If set, includes outer boundaries
/// @return Set of points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
   create(cell::Type celltype, int n, bool exterior);

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_warped(cell::Type celltype, int n, bool exterior);


}
}
