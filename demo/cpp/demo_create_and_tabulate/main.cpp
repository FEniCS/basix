// ==================================
// Creating and tabulating an element
// ==================================
//
// This demo shows how Basix can be used to create an element
// and tabulate the values of its basis functions at a set of
// points.

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <iostream>

namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

using T = double;

int main(int argc, char* argv[])
{
  // Create a degree 4 Lagrange element on a quadrilateral
  // For Lagrange elements, we use `basix::element::family::P`.
  auto family = basix::element::family::P;
  auto cell_type = basix::cell::type::quadrilateral;
  int k = 3;

  // For Lagrange elements, we must provide and extra argument: the Lagrange
  // variant. In this example, we use the equispaced variant: this will place
  // the degrees of freedom (DOFs) of the element in an equally spaced lattice.
  auto variant = basix::element::lagrange_variant::equispaced;

  // Create the lagrange element
  basix::FiniteElement lagrange = basix::create_element<T>(
      family, cell_type, k, variant, basix::element::dpc_variant::unset, false);

  // Get the number of degrees of freedom for the element
  int dofs = lagrange.dim();
  assert(dofs == (k + 1) * (k + 1));

  // Create a set of points, and tabulate the basis functions
  // of the Lagrange element at these points.
  std::vector<T> points = {0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.6, 0.4, 1.0};

  auto [tab_data, shape] = lagrange.tabulate(0, points, {points.size() / 2, 2});

  std::cout << "Tabulate data shape: [ ";
  for (auto s : shape)
    std::cout << s << " ";
  std::cout << "]" << std::endl;

  mdspan_t<const T, 4> tab(tab_data.data(), shape);
  std::cout << "Tabulate data (0, 0, :, 0): [ ";
  for (std::size_t i = 0; i < tab.extent(2); ++i)
    std::cout << tab[0, 0, i, 0] << " ";
  std::cout << "]" << std::endl;

  return 0;
}
