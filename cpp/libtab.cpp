
#include "libtab.h"
#include "cell.h"
#include "finite-element.h"
#include <memory>
#include <vector>

using namespace libtab;

std::vector<std::unique_ptr<FiniteElement>> _registry;

void check_handle(int handle)
{
  if (handle < 0 or handle >= (int)_registry.size())
    throw std::range_error("Bad index");
  if (!_registry[handle])
    throw std::runtime_error("Invalid element");
}

int libtab::register_element(const char* family_name, const char* cell_type,
                             int degree)
{
  _registry.push_back(std::make_unique<FiniteElement>(
      create_element(family_name, cell_type, degree)));
  return _registry.size() - 1;
}

void libtab::deregister_element(int handle)
{
  check_handle(handle);
  _registry[handle].reset();
}

std::vector<Eigen::ArrayXXd> libtab::tabulate(int handle, int nd,
                                              const Eigen::ArrayXXd& x)
{
  check_handle(handle);
  return _registry[handle]->tabulate(nd, x);
}

const char* libtab::cell_type(int handle)
{
  check_handle(handle);
  return cell::type_to_str(_registry[handle]->cell_type()).c_str();
}

int libtab::degree(int handle)
{
  check_handle(handle);
  return _registry[handle]->degree();
}

int libtab::value_size(int handle)
{
  check_handle(handle);
  return _registry[handle]->value_size();
}

const std::vector<int>& libtab::value_shape(int handle)
{
  check_handle(handle);
  return _registry[handle]->value_shape();
}

const Eigen::ArrayXXd& libtab::points(int handle)
{
  check_handle(handle);
  return _registry[handle]->points();
}

const Eigen::MatrixXd& libtab::interpolation_matrix(int handle)
{
  check_handle(handle);
  return _registry[handle]->interpolation_matrix();
}

const std::vector<std::vector<int>>& libtab::entity_dofs(int handle)
{
  check_handle(handle);
  return _registry[handle]->entity_dofs();
}

const char* libtab::family_name(int handle)
{
  check_handle(handle);
  return _registry[handle]->family_name().c_str();
}

const char* libtab::mapping_name(int handle)
{
  check_handle(handle);
  return _registry[handle]->mapping_name().c_str();
}

Eigen::ArrayXXd libtab::geometry(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::geometry(ct);
}

std::vector<std::vector<std::vector<int>>>
libtab::topology(const char* cell_type)
{
  cell::type ct = cell::str_to_type(cell_type);
  return cell::topology(ct);
}
