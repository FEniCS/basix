#pragma once

#include "cell.h"
#include <Eigen/Dense>
#include <vector>

namespace libtab
{
namespace lattice
{
enum class Type
{
  equispaced,
  gll_warped
};

/// Create a lattice of points on the cell, equispaced along each edge
/// optionally including the outer surface points
/// @param celltype The cell::Type
/// @param n Number of points in each direction
/// @param type Either lattice::Type::equispaced or lattice::Type::gll_warped
/// @param exterior If set, includes outer boundaries
/// @return Set of points
Eigen::ArrayXXd create(cell::Type celltype, int n, Type type, bool exterior);

} // namespace lattice
} // namespace libtab
