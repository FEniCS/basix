// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <iostream>

#include "docs.h"
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/indexing.h>
#include <basix/interpolation.h>
#include <basix/lattice.h>
#include <basix/maps.h>
#include <basix/polynomials.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/tensor.h>

#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace nb = nanobind;
using namespace basix;

namespace
{
const std::string& cell_type_to_str(cell::type type)
{
  static const std::map<cell::type, std::string> type_to_name
      = {{cell::type::point, "point"},
         {cell::type::interval, "interval"},
         {cell::type::triangle, "triangle"},
         {cell::type::tetrahedron, "tetrahedron"},
         {cell::type::quadrilateral, "quadrilateral"},
         {cell::type::pyramid, "pyramid"},
         {cell::type::prism, "prism"},
         {cell::type::hexahedron, "hexahedron"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}

template <typename T>
auto adapt_x(const T& x)
{
  std::size_t xsize = 1;
  std::vector<std::size_t> shape;
  for (std::size_t i = 0; i < x.ndim(); ++i)
  {
    shape.push_back(x.shape(i));
    xsize *= x.shape(i);
  }
  return xt::adapt(static_cast<const double*>(x.data()), xsize,
                   xt::no_ownership(), shape);
}

auto adapt_x(const nb::tensor<nb::numpy, double>& x)
{
  std::vector<std::size_t> shape;
  std::size_t size = 1;
  for (std::size_t i = 0; i < x.ndim(); ++i)
  {
    shape.push_back(x.shape(i));
    size *= x.shape(i);
  }
  return xt::adapt(static_cast<const double*>(x.data()), size,
                   xt::no_ownership(), shape);
}

/// Create a nb::tensor that shares data with an
/// xtensor array. The C++ object owns the data, and the
/// nb::tensor object keeps the C++ object alive.
// Similar to https://github.com/pybind/pybind11/issues/1042
template <typename shape_t, typename U>
auto xt_as_nbtensor(U&& x)
{
  std::vector<std::size_t> shape;
  std::size_t dim = 1;
  if constexpr (std::is_same<std::vector<typename U::value_type>, U>())
    shape = {x.size()};
  else
  {
    dim = x.dimension();
    shape.resize(dim);
    std::copy(x.shape().data(), x.shape().data() + dim, shape.begin());
  }
  auto data = x.data();
  auto size = x.size();
  std::unique_ptr<U> x_ptr = std::make_unique<U>(std::move(x));
  auto capsule = nb::capsule(x_ptr.get(), [](void* p) noexcept
                             { std::unique_ptr<U>(reinterpret_cast<U*>(p)); });
  x_ptr.release();
  return nb::tensor<nb::numpy, typename U::value_type, shape_t>(
      data, dim, shape.data(), capsule);
}

} // namespace

NB_MODULE(_basixcpp, m)
{
  //   m.doc() = R"(
  // Interface to the Basix C++ library.
  // )";

  m.attr("__version__") = basix::version();

  m.def("topology", &cell::topology, basix::docstring::topology.c_str());
  m.def(
      "geometry",
      [](cell::type celltype)
      {
        xt::xtensor<double, 2> g = cell::geometry(celltype);
        std::array<std::size_t, 2> shape = {g.shape(0), g.shape(1)};
        return nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
            g.data(), shape.size(), shape.data());
      },
      basix::docstring::geometry.c_str());
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity,
        basix::docstring::sub_entity_connectivity.c_str());
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index)
      {
        xt::xtensor<double, 2> g
            = cell::sub_entity_geometry(celltype, dim, index);
        std::array<std::size_t, 2> shape = {g.shape(0), g.shape(1)};
        return nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
            g.data(), shape.size(), shape.data());
      },
      basix::docstring::sub_entity_geometry.c_str());

  nb::enum_<lattice::type>(m, "LatticeType")
      .value("equispaced", lattice::type::equispaced)
      .value("gll", lattice::type::gll)
      .value("chebyshev", lattice::type::chebyshev)
      .value("gl", lattice::type::gl);
  nb::enum_<lattice::simplex_method>(m, "LatticeSimplexMethod")
      .value("none", lattice::simplex_method::none)
      .value("warp", lattice::simplex_method::warp)
      .value("isaac", lattice::simplex_method::isaac)
      .value("centroid", lattice::simplex_method::centroid);
  nb::enum_<polynomials::type>(m, "PolynomialType")
      .value("legendre", polynomials::type::legendre);

  m.def(
      "tabulate_polynomials",
      [](polynomials::type polytype, cell::type celltype, int d,
         nb::tensor<double> x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        std::array<std::size_t, 2> shape
            = {(std::size_t)x.shape(0), (std::size_t)x.shape(1)};
        auto _x = xt::adapt(static_cast<const double*>(x.data()),
                            (std::size_t)(shape[0] * shape[1]),
                            xt::no_ownership(), shape);
        xt::xtensor<double, 2> t
            = polynomials::tabulate(polytype, celltype, d, _x);

        return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(t));
      },
      basix::docstring::tabulate_polynomials.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior)
      {
        xt::xtensor<double, 2> l = lattice::create(
            celltype, n, type, exterior, lattice::simplex_method::none);
        return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(l));
      },
      basix::docstring::create_lattice__celltype_n_type_exterior.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method)
      {
        xt::xtensor<double, 2> l
            = lattice::create(celltype, n, type, exterior, method);
        return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(l));
      },
      basix::docstring::create_lattice__celltype_n_type_exterior_method
          .c_str());

  nb::enum_<maps::type>(m, "MapType")
      .value("identity", maps::type::identity)
      .value("L2Piola", maps::type::L2Piola)
      .value("covariantPiola", maps::type::covariantPiola)
      .value("contravariantPiola", maps::type::contravariantPiola)
      .value("doubleCovariantPiola", maps::type::doubleCovariantPiola)
      .value("doubleContravariantPiola", maps::type::doubleContravariantPiola);

  nb::enum_<quadrature::type>(m, "QuadratureType")
      .value("Default", quadrature::type::Default)
      .value("gauss_jacobi", quadrature::type::gauss_jacobi)
      .value("gll", quadrature::type::gll)
      .value("xiao_gimbutas", quadrature::type::xiao_gimbutas);

  nb::enum_<cell::type>(m, "CellType")
      .value("point", cell::type::point)
      .value("interval", cell::type::interval)
      .value("triangle", cell::type::triangle)
      .value("tetrahedron", cell::type::tetrahedron)
      .value("quadrilateral", cell::type::quadrilateral)
      .value("hexahedron", cell::type::hexahedron)
      .value("prism", cell::type::prism)
      .value("pyramid", cell::type::pyramid);

  m.def(
      "cell_volume",
      [](cell::type cell_type) -> double { return cell::volume(cell_type); },
      basix::docstring::cell_volume.c_str());
  m.def(
      "cell_facet_normals",
      [](cell::type cell_type)
      {
        xt::xtensor<double, 2> n = cell::facet_normals(cell_type);
        return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(n));
      },
      basix::docstring::cell_facet_normals.c_str());
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type)
      {
        xt::xtensor<double, 1> v = cell::facet_reference_volumes(cell_type);
        return xt_as_nbtensor<nb::shape<nb::any>>(std::move(v));
      },
      basix::docstring::cell_facet_reference_volumes.c_str());
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type)
      {
        xt::xtensor<double, 2> n = cell::facet_outward_normals(cell_type);
        return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(n));
      },
      basix::docstring::cell_facet_outward_normals.c_str());
  m.def(
      "cell_facet_orientations",
      [](cell::type cell_type)
      {
        std::vector<bool> c = cell::facet_orientations(cell_type);
        std::vector<std::uint8_t> c8(c.begin(), c.end());
        return c8;
      },
      basix::docstring::cell_facet_orientations.c_str());
  m.def(
      "cell_facet_jacobians",
      [](cell::type cell_type)
      {
        xt::xtensor<double, 3> J = cell::facet_jacobians(cell_type);
        return xt_as_nbtensor<nb::shape<nb::any, nb::any, nb::any>>(
            std::move(J));
      },
      basix::docstring::cell_facet_jacobians.c_str());

  nb::enum_<element::family>(m, "ElementFamily")
      .value("custom", element::family::custom)
      .value("P", element::family::P)
      .value("BDM", element::family::BDM)
      .value("RT", element::family::RT)
      .value("N1E", element::family::N1E)
      .value("N2E", element::family::N2E)
      .value("Regge", element::family::Regge)
      .value("HHJ", element::family::HHJ)
      .value("bubble", element::family::bubble)
      .value("serendipity", element::family::serendipity)
      .value("DPC", element::family::DPC)
      .value("CR", element::family::CR);

  nb::class_<FiniteElement>(m, "FiniteElement")
      .def(
          "tabulate",
          [](const FiniteElement& self, int n,
             const nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>,
                              nb::c_contig>& x)
          {
            std::vector<std::size_t> shape
                = {(std::size_t)x.shape(0), (std::size_t)x.shape(1)};
            auto _x = xt::adapt((double*)x.data(), shape[0] * shape[1],
                                xt::no_ownership(), shape);

            xt::xtensor<double, 4> t = self.tabulate(n, _x);
            return xt_as_nbtensor<
                nb::shape<nb::any, nb::any, nb::any, nb::any>>(std::move(t));
          },
          basix::docstring::FiniteElement__tabulate.c_str())
      .def("__eq__", &FiniteElement::operator==)
      .def(
          "push_forward",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any, nb::any, nb::any>,
                        nb::c_contig>
                 U,
             nb::tensor<double, nb::shape<nb::any, nb::any, nb::any>,
                        nb::c_contig>
                 J,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> detJ,
             nb::tensor<double, nb::shape<nb::any, nb::any, nb::any>,
                        nb::c_contig>
                 K)
          {
            xt::xtensor<double, 3> u = self.push_forward(
                adapt_x(U), adapt_x(J),
                xtl::span<const double>(static_cast<double*>(detJ.data()),
                                        detJ.shape(0)),
                adapt_x(K));
            return xt_as_nbtensor<nb::shape<nb::any, nb::any, nb::any>>(
                std::move(u));
          },
          basix::docstring::FiniteElement__push_forward.c_str())
      .def(
          "pull_back",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any, nb::any, nb::any>,
                        nb::c_contig>
                 u,
             nb::tensor<double, nb::shape<nb::any, nb::any, nb::any>,
                        nb::c_contig>
                 J,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> detJ,
             nb::tensor<double, nb::shape<nb::any, nb::any, nb::any>,
                        nb::c_contig>
                 K)
          {
            xt::xtensor<double, 3> U = self.pull_back(
                adapt_x(u), adapt_x(J),
                xtl::span<const double>(static_cast<double*>(detJ.data()),
                                        detJ.shape(0)),
                adapt_x(K));
            return xt_as_nbtensor<
                nb::shape<nb::any, nb::any, nb::any, nb::any>>(std::move(U));
          },
          basix::docstring::FiniteElement__pull_back.c_str())
      .def(
          "apply_dof_transformation",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            xtl::span<double> data_span(static_cast<double*>(data.data()),
                                        data.shape(0));
            self.apply_dof_transformation(data_span, block_size, cell_info);

            std::size_t size = data.shape(0);
            return nb::tensor<nb::numpy, double, nb::shape<nb::any>>(
                data_span.data(), 1, &size);
          },
          basix::docstring::FiniteElement__apply_dof_transformation.c_str())
      .def(
          "apply_dof_transformation_to_transpose",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            xtl::span<double> data_span(static_cast<double*>(data.data()),
                                        data.shape(0));
            self.apply_dof_transformation_to_transpose(data_span, block_size,
                                                       cell_info);
            std::size_t size = data.shape(0);
            nb::tensor<nb::numpy, double, nb::shape<nb::any>>(data_span.data(),
                                                              1, &size);
          },
          basix::docstring::FiniteElement__apply_dof_transformation_to_transpose
              .c_str())
      .def(
          "apply_inverse_transpose_dof_transformation",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            xtl::span<double> data_span(static_cast<double*>(data.data()),
                                        data.shape(0));
            self.apply_inverse_transpose_dof_transformation(
                data_span, block_size, cell_info);
            std::size_t size = data.shape(0);
            return nb::tensor<nb::numpy, double, nb::shape<nb::any>>(
                data_span.data(), 1, &size);
          },
          basix::docstring::
              FiniteElement__apply_inverse_transpose_dof_transformation.c_str())
      .def(
          "base_transformations",
          [](const FiniteElement& self)
          {
            xt::xtensor<double, 3> t = self.base_transformations();
            return xt_as_nbtensor<nb::shape<nb::any, nb::any, nb::any>>(
                std::move(t));
          },
          basix::docstring::FiniteElement__base_transformations.c_str())
      .def(
          "entity_transformations",
          [](const FiniteElement& self)
          {
            std::map<cell::type, xt::xtensor<double, 3>> t
                = self.entity_transformations();
            nb::dict t2;
            for (auto tpart : t)
            {
              std::array<std::size_t, 3> shape
                  = {tpart.second.shape(0), tpart.second.shape(1),
                     tpart.second.shape(2)};
              t2[cell_type_to_str(tpart.first).c_str()]
                  = nb::tensor<nb::numpy, double,
                               nb::shape<nb::any, nb::any, nb::any>>(
                      tpart.second.data(), shape.size(), shape.data());
            }
            return t2;
          },
          basix::docstring::FiniteElement__entity_transformations.c_str())
      .def(
          "get_tensor_product_representation",
          [](const FiniteElement& self)
          {
            auto w = self.get_tensor_product_representation();
            nb::list l;
            for (auto v : w)
            {
              nb::list a_list;
              a_list.append(std::get<0>(v));
              a_list.append(std::get<1>(v));
              l.append(a_list);
            }
            return l;
          },
          basix::docstring::FiniteElement__get_tensor_product_representation
              .c_str())
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("highest_degree", &FiniteElement::highest_degree)
      .def_property_readonly("highest_complete_degree",
                             &FiniteElement::highest_complete_degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("dim", &FiniteElement::dim)
      .def_property_readonly("num_entity_dofs", &FiniteElement::num_entity_dofs)
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("num_entity_closure_dofs",
                             &FiniteElement::num_entity_closure_dofs)
      .def_property_readonly("entity_closure_dofs",
                             &FiniteElement::entity_closure_dofs)
      .def_property_readonly("value_size",
                             [](const FiniteElement& self)
                             {
                               return std::accumulate(
                                   self.value_shape().begin(),
                                   self.value_shape().end(), 1,
                                   std::multiplies<int>());
                             })
      .def_property_readonly("value_shape", &FiniteElement::value_shape)
      .def_property_readonly("discontinuous", &FiniteElement::discontinuous)
      .def_property_readonly("family", &FiniteElement::family)
      .def_property_readonly("lagrange_variant",
                             &FiniteElement::lagrange_variant)
      .def_property_readonly("dpc_variant", &FiniteElement::dpc_variant)
      .def_property_readonly(
          "dof_transformations_are_permutations",
          &FiniteElement::dof_transformations_are_permutations)
      .def_property_readonly("dof_transformations_are_identity",
                             &FiniteElement::dof_transformations_are_identity)
      .def_property_readonly("interpolation_is_identity",
                             &FiniteElement::interpolation_is_identity)
      .def_property_readonly("map_type", &FiniteElement::map_type)
      .def_property_readonly(
          "points",
          [](const FiniteElement& self)
          {
            // const xt::xtensor<double, 2>& x =
            // self.points();
            const xt::xtensor<double, 2>& x = self.points();
            std::array<std::size_t, 2> shape = {x.shape(0), x.shape(1)};
            return nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
                const_cast<double*>(x.data()), shape.size(), shape.data(),
                nb::cast(self));
          })
      .def_property_readonly(
          "interpolation_matrix",
          [](const FiniteElement& self)
          {
            // const xt::xtensor<double, 2>& P =
            // self.interpolation_matrix();
            xt::xtensor<double, 2> P = self.interpolation_matrix();
            return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(P));
          })
      .def_property_readonly(
          "dual_matrix",
          [](const FiniteElement& self)
          {
            // const xt::xtensor<double, 2>& P = self.dual_matrix();
            xt::xtensor<double, 2> P = self.dual_matrix();
            return xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(P));
          })
      .def_property_readonly(
          "coefficient_matrix",
          [](const FiniteElement& self)
          {
            // const xt::xtensor<double, 2>& P =
            // self.coefficient_matrix();
            xt::xtensor<double, 2> P = self.coefficient_matrix();
            std::array<std::size_t, 2> shape = {P.shape(0), P.shape(1)};
            return nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
                P.data(), shape.size(), shape.data());
          })
      .def_property_readonly(
          "wcoeffs",
          [](const FiniteElement& self)
          {
            //  const xt::xtensor<double, 2>& P =
            //  self.wcoeffs();
            xt::xtensor<double, 2> P = self.wcoeffs();
            std::array<std::size_t, 2> shape = {P.shape(0), P.shape(1)};
            return nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
                P.data(), shape.size(), shape.data());
          })
      .def_property_readonly(
          "M",
          [](const FiniteElement& self)
          {
            // const
            // std::array<std::vector<xt::xtensor<double,
            // 3>>, 4>& _M
            //     = self.M();
            // std::vector<std::vector<py::array_t<double,
            // py::array::c_style>>> M(
            //     4);
            std::array<std::vector<xt::xtensor<double, 3>>, 4> _M = self.M();
            std::vector<std::vector<nb::tensor<
                nb::numpy, double, nb::shape<nb::any, nb::any, nb::any>>>>
                M(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _M[i].size(); ++j)
              {
                // M[i].push_back(py::array_t<double>(
                //     _M[i][j].shape(), _M[i][j].data(),
                //     py::cast(self)));
                std::array<std::size_t, 3> shape
                    = {_M[i][j].shape(0), _M[i][j].shape(1), _M[i][j].shape(2)};
                M[i].push_back(nb::tensor<nb::numpy, double,
                                          nb::shape<nb::any, nb::any, nb::any>>(
                    _M[i][j].data(), shape.size(), shape.data()));
              }
            }
            return M;
          })
      .def_property_readonly(
          "x",
          [](const FiniteElement& self)
          {
            // const
            // std::array<std::vector<xt::xtensor<double,
            // 2>>, 4>& _x
            //     = self.x();
            std::array<std::vector<xt::xtensor<double, 2>>, 4> _x = self.x();
            // std::vector<std::vector<py::array_t<double,
            // py::array::c_style>>> x(
            //     4);
            std::vector<std::vector<
                nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>>>
                x(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _x[i].size(); ++j)
              {
                // x[i].push_back(py::array_t<double>(
                //     _x[i][j].shape(), _x[i][j].data(),
                //     py::cast(self)));
                std::array<std::size_t, 2> shape
                    = {_x[i][j].shape(0), _x[i][j].shape(1)};
                x[i].push_back(
                    nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
                        _x[i][j].data(), shape.size(), shape.data()));
              }
            }
            return x;
          })
      .def_property_readonly("has_tensor_product_factorisation",
                             &FiniteElement::has_tensor_product_factorisation);

  nb::enum_<element::lagrange_variant>(m, "LagrangeVariant")
      .value("unset", element::lagrange_variant::unset)
      .value("equispaced", element::lagrange_variant::equispaced)
      .value("gll_warped", element::lagrange_variant::gll_warped)
      .value("gll_isaac", element::lagrange_variant::gll_isaac)
      .value("gll_centroid", element::lagrange_variant::gll_centroid)
      .value("chebyshev_warped", element::lagrange_variant::chebyshev_warped)
      .value("chebyshev_isaac", element::lagrange_variant::chebyshev_isaac)
      .value("chebyshev_centroid",
             element::lagrange_variant::chebyshev_centroid)
      .value("gl_warped", element::lagrange_variant::gl_warped)
      .value("gl_isaac", element::lagrange_variant::gl_isaac)
      .value("gl_centroid", element::lagrange_variant::gl_centroid)
      .value("legendre", element::lagrange_variant::legendre)
      .value("vtk", element::lagrange_variant::vtk);

  nb::enum_<element::dpc_variant>(m, "DPCVariant")
      .value("unset", element::dpc_variant::unset)
      .value("simplex_equispaced", element::dpc_variant::simplex_equispaced)
      .value("simplex_gll", element::dpc_variant::simplex_gll)
      .value("horizontal_equispaced",
             element::dpc_variant::horizontal_equispaced)
      .value("horizontal_gll", element::dpc_variant::horizontal_gll)
      .value("diagonal_equispaced", element::dpc_variant::diagonal_equispaced)
      .value("diagonal_gll", element::dpc_variant::diagonal_gll)
      .value("legendre", element::dpc_variant::legendre);

  // Create FiniteElement
  m.def(
      "create_custom_element",
      [](cell::type cell_type, const std::vector<int>& value_shape,
         const nb::tensor<nb::numpy, double>& wcoeffs,
         const std::vector<std::vector<nb::tensor<nb::numpy, double>>>& x,
         const std::vector<std::vector<nb::tensor<nb::numpy, double>>>& M,
         maps::type map_type, bool discontinuous, int highest_complete_degree,
         int highest_degree) -> FiniteElement
      {
        if (x.size() != 4)
          throw std::runtime_error("x has the wrong size");
        if (M.size() != 4)
          throw std::runtime_error("M has the wrong size");
        xt::xtensor<double, 2> _wco = adapt_x(wcoeffs);

        std::array<std::vector<xt::xtensor<double, 2>>, 4> _x;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < x[i].size(); ++j)
          {
            if (x[i][j].ndim() != 2)
              throw std::runtime_error("Incorrect dim in x");
            _x[i].push_back(adapt_x(x[i][j]));
          }
        }

        std::array<std::vector<xt::xtensor<double, 3>>, 4> _M;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < M[i].size(); ++j)
          {
            if (M[i][j].ndim() != 3)
              throw std::runtime_error("Incorrect dim in M");
            _M[i].push_back(adapt_x(M[i][j]));
          }
        }

        std::vector<std::size_t> _vs(value_shape.size());
        for (std::size_t i = 0; i < value_shape.size(); ++i)
          _vs[i] = static_cast<std::size_t>(value_shape[i]);

        return basix::create_custom_element(
            cell_type, _vs, _wco, _x, _M, map_type, discontinuous,
            highest_complete_degree, highest_degree);
      },
      basix::docstring::create_custom_element.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         bool discontinuous) -> FiniteElement
      {
        return basix::create_element(family_name, cell_name, degree,
                                     discontinuous);
      },
      basix::docstring::create_element__family_cell_degree_discontinuous
          .c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant lvariant,
         bool discontinuous) -> FiniteElement
      {
        return basix::create_element(family_name, cell_name, degree, lvariant,
                                     discontinuous);
      },
      basix::docstring::
          create_element__family_cell_degree_lvariant_discontinuous.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::dpc_variant dvariant, bool discontinuous) -> FiniteElement
      {
        return basix::create_element(family_name, cell_name, degree, dvariant,
                                     discontinuous);
      },
      basix::docstring::
          create_element__family_cell_degree_dvariant_discontinuous.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant lvariant, element::dpc_variant dvariant,
         bool discontinuous) -> FiniteElement
      {
        return basix::create_element(family_name, cell_name, degree, lvariant,
                                     dvariant, discontinuous);
      },
      basix::docstring::
          create_element__family_cell_degree_lvariant_dvariant_discontinuous
              .c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name,
         int degree) -> FiniteElement
      { return basix::create_element(family_name, cell_name, degree); },
      basix::docstring::create_element__family_cell_degree.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant lvariant) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree, lvariant);
      },
      basix::docstring::create_element__family_cell_degree_lvariant.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::dpc_variant dvariant) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree, dvariant);
      },
      basix::docstring::create_element__family_cell_degree_dvariant.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant lvariant,
         element::dpc_variant dvariant) -> FiniteElement
      {
        return basix::create_element(family_name, cell_name, degree, lvariant,
                                     dvariant);
      },
      basix::docstring::create_element__family_cell_degree_lvariant_dvariant
          .c_str());

  // Interpolate between elements
  m.def(
      "compute_interpolation_operator",
      [](const FiniteElement& element_from, const FiniteElement& element_to)
      // -> const py::array_t<double, py::array::c_style>
      {
        xt::xtensor<double, 2> out
            = basix::compute_interpolation_operator(element_from, element_to);
        std::array<std::size_t, 2> shape = {out.shape(0), out.shape(1)};
        return nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>(
            out.data(), shape.size(), shape.data());
        // return py::array_t<double>(out.shape(), out.data());
      },
      basix::docstring::compute_interpolation_operator.c_str());

  m.def(
      "tabulate_polynomial_set",
      [](cell::type celltype, int d, int n,
         const nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>& x)
      {
        std::vector<std::size_t> shape;
        std::size_t xsize = 1;
        for (std::size_t i = 0; i < x.ndim(); ++i)
        {
          shape.push_back(x.shape(i));
          xsize *= x.shape(i);
        }
        auto _x = xt::adapt(static_cast<const double*>(x.data()), xsize,
                            xt::no_ownership(), shape);
        xt::xtensor<double, 3> P = polyset::tabulate(celltype, d, n, _x);
        return xt_as_nbtensor<nb::shape<nb::any, nb::any, nb::any>>(
            std::move(P));
      },
      basix::docstring::tabulate_polynomial_set.c_str());

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, int m)
      {
        auto [pts, w] = quadrature::make_quadrature(rule, celltype, m);
        // FIXME: it would be more elegant to handle 1D case as a 1D
        // array, but FFCx would need updating

        auto pt = xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(pts));
        auto wt = xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(w));

        nb::list l;
        l.append(pt);
        l.append(wt);
        return l;
      },
      basix::docstring::make_quadrature__rule_celltype_m.c_str());

  m.def(
      "make_quadrature",
      [](cell::type celltype, int m)
      {
        auto [pts, w] = quadrature::make_quadrature(celltype, m);
        // FIXME: it would be more elegant to handle 1D case as a 1D
        // array, but FFCx would need updating

        auto pt = xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(pts));
        auto wt = xt_as_nbtensor<nb::shape<nb::any, nb::any>>(std::move(w));

        nb::list l;
        l.append(pt);
        l.append(wt);
        return l;
      },
      basix::docstring::make_quadrature__celltype_m.c_str());

  m.def("index", nb::overload_cast<int>(&basix::indexing::idx),
        basix::docstring::index__p.c_str())
      .def("index", nb::overload_cast<int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q.c_str())
      .def("index", nb::overload_cast<int, int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q_r.c_str());
}
