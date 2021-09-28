#include <memory>

#include <basix/finite-element.h>

int main()
{
  auto _element = std::make_unique<basix::FiniteElement>(basix::create_element(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::equispaced, false));

  return 0;
}
