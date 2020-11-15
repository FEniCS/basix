#include <Eigen/Dense>
#include <vector>
#include "cell.h"

#pragma once

namespace libtab
{
namespace lattice
{
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
   create(cell::Type celltype, int n, bool exterior);

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_warped(cell::Type celltype, int n, bool exterior);


}
}
