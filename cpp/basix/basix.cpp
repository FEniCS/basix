// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "basix.h"
#include "cell.h"
#include "finite-element.h"
#include "mappings.h"
#include "quadrature.h"
#include "span.hpp"
#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <xtensor/xio.hpp>

using namespace basix;

std::vector<std::unique_ptr<FiniteElement>> _registry;

void check_handle(int handle)
{
  if (handle < 0 or handle >= (int)_registry.size())
    throw std::range_error("Bad index");
  if (!_registry[handle])
    throw std::runtime_error("Invalid element");
}
namespace

{
template <typename T>
void _map_push_forward(int handle, T* u, const T* U, const double* J,
                       const double* detJ, const double* K,
                       const int physical_dim,
                       const int /*physical_value_size*/, const int nresults,
                       const int num_points)
{
  check_handle(handle);
  const std::size_t tdim
      = cell::topological_dimension(_registry[handle]->cell_type());
  const std::size_t vs = _registry[handle]->value_size();
  std::array<std::size_t, 3> s0
      = {(std::size_t)num_points, (std::size_t)nresults, vs};
  auto _U = xt::adapt(U, s0[0] * s0[1] * s0[2], xt::no_ownership(), s0);
  std::array<std::size_t, 3> s1
      = {(std::size_t)num_points, (std::size_t)physical_dim, tdim};
  auto _J = xt::adapt(J, s1[0] * s1[1] * s1[2], xt::no_ownership(), s1);
  std::array<std::size_t, 3> s2
      = {(std::size_t)num_points, tdim, (std::size_t)physical_dim};
  auto _K = xt::adapt(K, s2[0] * s2[1] * s2[2], xt::no_ownership(), s2);
  auto _detJ = tcb::span(detJ, num_points);

  auto _u = xt::adapt(u, s0[0] * s0[1] * s0[2], xt::no_ownership(), s0);
  _registry[handle]->map_push_forward_m<T>(_U, _J, _detJ, _K, _u);
}

template <typename T>
void _map_pull_back(int handle, T* U, const T* u, const double* J,
                    const double* detJ, const double* K, const int physical_dim,
                    const int physical_value_size, const int nresults,
                    const int num_points)
{
  // FIXME: need to sort out row-major column major storage and expected
  // input layout. It does really weird things to interface with DOLFIN.

  check_handle(handle);
  const std::size_t tdim
      = cell::topological_dimension(_registry[handle]->cell_type());

  std::array<std::size_t, 3> s0
      = {(std::size_t)physical_value_size, (std::size_t)nresults,
         (std::size_t)num_points};
  auto _u = xt::adapt(u, s0[0] * s0[1] * s0[2], xt::no_ownership(), s0);

  std::array<std::size_t, 3> s1
      = {(std::size_t)num_points, (std::size_t)physical_dim, tdim};
  auto _J = xt::adapt(J, s1[0] * s1[1] * s1[2], xt::no_ownership(), s1);

  std::array<std::size_t, 3> s2
      = {(std::size_t)num_points, tdim, (std::size_t)physical_dim};
  auto _K = xt::adapt(K, s2[0] * s2[1] * s2[2], xt::no_ownership(), s2);
  auto _detJ = tcb::span(detJ, num_points);

  xt::xtensor<T, 3> u_t = xt::transpose(_u);

  std::array<std::size_t, 3> s3
      = {(std::size_t)num_points, (std::size_t)nresults,
         (std::size_t)physical_value_size};
  xt::xtensor<T, 3> _U(s3);
  _registry[handle]->map_pull_back_m<T>(u_t, _J, _detJ, _K, _U);

  auto tmp = xt::adapt(U, s0[0] * s0[1] * s0[2], xt::no_ownership(), s0);
  tmp = xt::transpose(_U);
}
} // namespace

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
                     int npoints)
{
  check_handle(handle);

  // gdim and tdim are the same for all cells in basix
  const std::size_t gdim
      = cell::topological_dimension(_registry[handle]->cell_type());
  xt::xarray<int>::shape_type s({(std::size_t)npoints, gdim});
  auto _x = xt::adapt(x, npoints * gdim, xt::no_ownership(), s);

  _registry[handle]->tabulate(nd, _x, basis_values);
}

void basix::map_push_forward_real(int handle, double* physical_data,
                                  const double* reference_data, const double* J,
                                  const double* detJ, const double* K,
                                  const int physical_dim,
                                  const int physical_value_size,
                                  const int nresults, const int npoints)
{
  _map_push_forward<double>(handle, physical_data, reference_data, J, detJ, K,
                            physical_dim, physical_value_size, nresults,
                            npoints);
}

void basix::map_pull_back_real(int handle, double* reference_data,
                               const double* physical_data, const double* J,
                               const double* detJ, const double* K,
                               const int physical_dim,
                               const int physical_value_size,
                               const int nresults, const int npoints)
{
  _map_pull_back<double>(handle, reference_data, physical_data, J, detJ, K,
                         physical_dim, physical_value_size, nresults, npoints);
}

void basix::map_push_forward_complex(int handle,
                                     std::complex<double>* physical_data,
                                     const std::complex<double>* reference_data,
                                     const double* J, const double* detJ,
                                     const double* K, const int physical_dim,
                                     const int physical_value_size,
                                     const int nresults, const int npoints)
{
  _map_push_forward<std::complex<double>>(
      handle, physical_data, reference_data, J, detJ, K, physical_dim,
      physical_value_size, nresults, npoints);
}

void basix::map_pull_back_complex(int handle,
                                  std::complex<double>* reference_data,
                                  const std::complex<double>* physical_data,
                                  const double* J, const double* detJ,
                                  const double* K, const int physical_dim,
                                  const int physical_value_size,
                                  const int nresults, const int npoints)
{
  _map_pull_back<std::complex<double>>(handle, reference_data, physical_data, J,
                                       detJ, K, physical_dim,
                                       physical_value_size, nresults, npoints);
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

int basix::dim(int handle)
{
  check_handle(handle);
  return _registry[handle]->dim();
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

int basix::interpolation_num_points(int handle)
{
  check_handle(handle);
  return _registry[handle]->num_points();
}

void basix::interpolation_points(int handle, double* points)
{
  check_handle(handle);
  std::size_t num_rows = interpolation_num_points(handle);
  std::size_t num_cols = cell_geometry_dimension(cell_type(handle));
  xt::xarray<int>::shape_type s({num_rows, num_cols});
  auto m = xt::adapt(points, num_rows * num_cols, xt::no_ownership(), s);
  m = _registry[handle]->points();
}

void basix::interpolation_matrix(int handle, double* matrix)
{
  check_handle(handle);
  std::size_t num_rows = dim(handle);
  std::size_t num_cols
      = interpolation_num_points(handle) * _registry[handle]->value_size();
  xt::xarray<int>::shape_type s({num_rows, num_cols});
  auto m = xt::adapt(matrix, num_rows * num_cols, xt::no_ownership(), s);
  m = _registry[handle]->interpolation_matrix();
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
  return cell::geometry(ct).shape()[0];
}

int basix::cell_geometry_dimension(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::geometry(ct).shape()[1];
}

void basix::cell_geometry(const char* cell_type, double* points)
{
  cell::type ct = cell::str_to_type(cell_type);
  xt::xtensor<double, 2> pts = cell::geometry(ct);
  std::copy(pts.data(), pts.data() + pts.size(), points);
}

std::vector<std::vector<std::vector<int>>>
basix::topology(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::topology(ct);
}
