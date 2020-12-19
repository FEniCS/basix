
#include "libtab.h"
#include "finite-element.h"
#include <memory>
#include <vector>

using namespace libtab;

std::vector<std::shared_ptr<FiniteElement>> _registry;

int libtab::register_element(std::string family_name, std::string cell_type,
                             int degree)
{
  _registry.push_back(std::make_shared<FiniteElement>(
      create_element(family_name, cell_type, degree)));
  return _registry.size() - 1;
}

void libtab::deregister_element(int index)
{
  if (index < 0 or index >= _registry.size())
    throw std::runtime_error("Bad index");

  _registry[index].reset();
}

std::vector<Eigen::ArrayXXd> tabulate(int handle, int nd,
                                      const Eigen::ArrayXXd& x)
{
  if (handle < 0 or handle >= _registry.size())
    throw std::runtime_error("Bad index");
  return _registry[handle]->tabulate(nd, x);
}
