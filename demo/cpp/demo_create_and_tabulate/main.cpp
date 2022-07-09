// ==================================
// Creating and tabulating an element
// ==================================
//
// This demo shows how Basix can be used to create an element
// and tabulate the values of its basis functions at a set of
// points.

#include <basix/finite-element.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

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
  basix::FiniteElement lagrange
      = basix::create_element(family, cell_type, k, variant);

  // Get the number of degrees of freedom for the element
  int dofs = lagrange.dim();
  assert(dofs == (k + 1) * (k + 1));

  // Create a set of points, and tabulate the basis functions
  // of the Lagrange element at these points.
  std::vector<double> points
      = {0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.6, 0.4, 1.0};

  xt::xtensor<double, 4> tab
      = lagrange.tabulate(0, points, {points.size() / 2, 2});

  std::cout << "\nTabulate data: \n"
            << xt::view(tab, 0, xt::all(), xt::all(), 0);
  std::cout << "\nTabulate data shape: " << xt::adapt(tab.shape());

  return 0;
}
