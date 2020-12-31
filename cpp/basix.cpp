
#include "basix.h"
#include "cell.h"
#include "finite-element.h"
#include "quadrature.h"
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

std::vector<Eigen::ArrayXXd> basix::tabulate(int handle, int nd,
                                             const Eigen::ArrayXXd& x)
{
  check_handle(handle);
  return _registry[handle]->tabulate(nd, x);
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

int basix::value_size(int handle)
{
  check_handle(handle);
  return _registry[handle]->value_size();
}

const std::vector<int>& basix::value_shape(int handle)
{
  check_handle(handle);
  return _registry[handle]->value_shape();
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

const std::vector<std::vector<int>>& basix::entity_dofs(int handle)
{
  check_handle(handle);
  return _registry[handle]->entity_dofs();
}

const char* basix::family_name(int handle)
{
  check_handle(handle);
  return _registry[handle]->family_name().c_str();
}

const char* basix::mapping_name(int handle)
{
  check_handle(handle);
  return _registry[handle]->mapping_name().c_str();
}

Eigen::ArrayXXd basix::geometry(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::geometry(ct);
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
