// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "basix.h"
#include "cell.h"
#include "finite-element.h"
#include "mappings.h"
#include "quadrature.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

using namespace basix;

std::vector<std::unique_ptr<FiniteElement>> _registry;

void check_handle(int handle)
{
  if (handle < 0 or handle >= (int)_registry.size())
    throw std::range_error("Bad index");
  if (!_registry[handle])
    throw std::runtime_error("Invalid element");
}

int basix::register_element(const char* family_name, const char* cell_type,
                            int degree)
{
  _registry.push_back(std::make_unique<FiniteElement>(
      create_element(family_name, cell_type, degree)));
  return _registry.size() - 1;
}

void basix::release_element(int handle)
{
  check_handle(handle);
  _registry[handle].reset();
  while (!_registry.empty() and !_registry.back())
    _registry.pop_back();
}

void basix::tabulate(int handle, double* basis_values, int nd, const double* x,
                     int npoints, int gdim)
{
  check_handle(handle);

  Eigen::Map<const Eigen::ArrayXXd> _x(x, npoints, gdim);
  std::vector<Eigen::ArrayXXd> values = _registry[handle]->tabulate(nd, _x);

  const int m = values[0].rows() * values[0].cols();
  for (std::size_t i = 0; i < values.size(); ++i)
    std::copy(values[i].data(), values[i].data() + m, basis_values + i * m);
}

Eigen::ArrayXXd basix::map_push_forward(int handle,
                                        const Eigen::ArrayXd& reference_data,
                                        const Eigen::MatrixXd& J, double detJ,
                                        const Eigen::MatrixXd& K)
{
  check_handle(handle);
  return _registry[handle]->map_push_forward(reference_data, J, detJ, K);
}

Eigen::ArrayXXd basix::map_pull_back(int handle,
                                     const Eigen::ArrayXd& physical_data,
                                     const Eigen::MatrixXd& J, double detJ,
                                     const Eigen::MatrixXd& K)
{
  check_handle(handle);
  return _registry[handle]->map_pull_back(physical_data, J, detJ, K);
}

const char* basix::cell_type(int handle)
{
  check_handle(handle);
  return cell::type_to_str(_registry[handle]->cell_type()).c_str();
}

int basix::degree(int handle)
{
  check_handle(handle);
  return _registry[handle]->degree();
}

int basix::value_rank(int handle)
{
  check_handle(handle);
  return _registry[handle]->value_shape().size();
}

void basix::value_shape(int handle, int* dimensions)
{
  check_handle(handle);
  std::vector<int> dims = _registry[handle]->value_shape();
  std::copy(dims.begin(), dims.end(), dimensions);
}

const Eigen::ArrayXXd& basix::points(int handle)
{
  check_handle(handle);
  return _registry[handle]->points();
}

const Eigen::MatrixXd& basix::interpolation_matrix(int handle)
{
  check_handle(handle);
  return _registry[handle]->interpolation_matrix();
}

void basix::entity_dofs(int handle, int dim, int* num_dofs)
{
  check_handle(handle);
  std::vector<std::vector<int>> dof_counts = _registry[handle]->entity_dofs();
  std::copy(dof_counts[dim].begin(), dof_counts[dim].end(), num_dofs);
}

const char* basix::family_name(int handle)
{
  check_handle(handle);
  return element::type_to_str(_registry[handle]->family()).c_str();
}

const char* basix::mapping_name(int handle)
{
  check_handle(handle);
  return mapping::type_to_str(_registry[handle]->mapping_type()).c_str();
}

int basix::cell_geometry_num_points(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::geometry(ct).rows();
}

int basix::cell_geometry_dimension(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::geometry(ct).cols();
}

void basix::geometry(const char* cell_type, double* points)
{
  cell::type ct = cell::str_to_type(cell_type);
  Eigen::ArrayXXd pts = cell::geometry(ct);
  std::copy(pts.data(), pts.data() + pts.rows() * pts.cols(), points);
}

std::vector<std::vector<std::vector<int>>>
basix::topology(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::topology(ct);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXd>
basix::make_quadrature(const char* rule, const char* cell_type, int order)
{
  cell::type ct = cell::str_to_type(cell_type);
  return basix::quadrature::make_quadrature(rule, ct, order);
}
