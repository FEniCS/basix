// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <basix/cell.h>
#include <basix/docs.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/indexing.h>
#include <basix/interpolation.h>
#include <basix/lattice.h>
#include <basix/maps.h>
#include <basix/mdspan.hpp>
#include <basix/polynomials.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/tensor.h>

#include <string>
#include <vector>
#include <span>

namespace nb = nanobind;
using namespace basix;

namespace stdex = std::experimental;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

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

// template <typename T>
// auto adapt_x(const T& x)
// {
//   std::size_t xsize = 1;
//   std::vector<std::size_t> shape;
//   for (std::size_t i = 0; i < x.ndim(); ++i)
//   {
//     shape.push_back(x.shape(i));
//     xsize *= x.shape(i);
//   }
//   return xt::adapt(static_cast<const double*>(x.data()), xsize,
//                    xt::no_ownership(), shape);
// }

// auto adapt_x(const nb::tensor<nb::numpy, double>& x)
// {
//   std::vector<std::size_t> shape;
//   std::size_t size = 1;
//   for (std::size_t i = 0; i < x.ndim(); ++i)
//   {
//     shape.push_back(x.shape(i));
//     size *= x.shape(i);
//   }
//   return xt::adapt(static_cast<const double*>(x.data()), size,
//                    xt::no_ownership(), shape);
// }

// /// Create a nb::tensor that shares data with an
// /// xtensor array. The C++ object owns the data, and the
// /// nb::tensor object keeps the C++ object alive.
// // Similar to https://github.com/pybind/pybind11/issues/1042
// template <typename T>
// auto as_nbtensor(std::vector<std::size_t>& shape, T* data)
// {
//   std::size_t dim = shape.size();
//   if constexpr (std::is_same<std::vector<typename U::value_type>, U>())
//     shape = {x.size()};
//   else
//   {
//     dim = x.dimension();
//     shape.resize(dim);
//     std::copy(x.shape().data(), x.shape().data() + dim, shape.begin());
//   }
//   auto data = x.data();
//   auto size = x.size();
//   std::unique_ptr<U> x_ptr = std::make_unique<U>(std::move(x));
//   auto capsule = nb::capsule(x_ptr.get(), [](void* p) noexcept
//                              { std::unique_ptr<U>(reinterpret_cast<U*>(p)); });
//   x_ptr.release();
//   return nb::tensor<nb::numpy, typename U::value_type, shape_t>(
//       data, dim, shape.data(), capsule);
// }

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
        auto [x, shape] = cell::geometry(celltype);
        return nb::tensor<nb::numpy, double>(x.data(), shape.size(), shape.data());
      },
      basix::docstring::geometry.c_str());
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity,
        basix::docstring::sub_entity_connectivity.c_str());
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index)
      {
        auto [x, shape] = cell::sub_entity_geometry(celltype, dim, index);
        return nb::tensor<nb::numpy, double>(x.data(), shape.size(), shape.data());
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
      .value("legendre", polynomials::type::legendre)
      .value("bernstein", polynomials::type::bernstein);

  m.def(
      "tabulate_polynomials",
      [](polynomials::type polytype, cell::type celltype, int d,
         nb::tensor<double> x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> _x(
            (double *)x.data(), x.shape(0), x.shape(1));
        auto [p, shape] = polynomials::tabulate(polytype, celltype, d, _x);
        return nb::tensor<double>(p.data(), shape.size(), shape.data());
      },
      basix::docstring::tabulate_polynomials.c_str());
  m.def("polynomials_dim", &polynomials::dim,
        basix::docstring::polynomials_dim.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior)
      {
        auto [x, shape] = lattice::create(celltype, n, type, exterior,
                                          lattice::simplex_method::none);
        return nb::tensor<nb::numpy, double>(x.data(), shape.size(), shape.data());
      },
      basix::docstring::create_lattice__celltype_n_type_exterior.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method)
      {
        auto [x, shape] = lattice::create(celltype, n, type, exterior, method);
        return nb::tensor<nb::numpy, double>(x.data(), shape.size(), shape.data());
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

  nb::enum_<cell::type>(m, "CellType", nb::is_arithmetic())
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
        auto [n, shape] = cell::facet_normals(cell_type);
        return nb::tensor<double>(n.data(), shape.size(), shape.data());
      },
      basix::docstring::cell_facet_normals.c_str());
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type)
      {
        std::vector<double> v = cell::facet_reference_volumes(cell_type);
        std::array<std::size_t, 1> shape = {v.size()};
        return nb::tensor<double>(v.data(), 1, shape.data());
      },
      basix::docstring::cell_facet_reference_volumes.c_str());
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type)
      {
        auto [n, shape] = cell::facet_outward_normals(cell_type);
        return nb::tensor<double>(n.data(), shape.size(), shape.data());
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
        auto [J, shape] = cell::facet_jacobians(cell_type);
        return nb::tensor<double>(J.data(), shape.size(), shape.data());
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
      .value("CR", element::family::CR)
      .value("Hermite", element::family::Hermite);

  nb::class_<FiniteElement>(m, "FiniteElement")
      .def(
          "tabulate",
          [](const FiniteElement& self, int n,
             const nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>&
                 x)
          {
            if (x.ndim() != 2)
              throw std::runtime_error("x has the wrong size");
            stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> _x(
                (const double *)x.data(), x.shape(0), x.shape(1));
            auto [t, shape] = self.tabulate(n, _x);
            return nb::tensor<double>(t.data(), shape.size(), shape.data());
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
            auto [u, shape] = self.push_forward(
                cmdspan3_t((double *)U.data(), U.shape(0), U.shape(1), U.shape(2)),
                cmdspan3_t((double *)J.data(), J.shape(0), J.shape(1), J.shape(2)),
                std::span<const double>((const double *)detJ.data(), detJ.shape(0)),
                cmdspan3_t((double *)K.data(), K.shape(0), K.shape(1), K.shape(2)));
            return nb::tensor<double>(u.data(), shape.size(), shape.data());
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
            auto [U, shape] = self.pull_back(
                cmdspan3_t((double *)u.data(), u.shape(0), u.shape(1), u.shape(2)),
                cmdspan3_t((double *)J.data(), J.shape(0), J.shape(1), J.shape(2)),
                std::span<const double>((const double *)detJ.data(), detJ.shape(0)),
                cmdspan3_t((double *)K.data(), K.shape(0), K.shape(1), K.shape(2)));
            return nb::tensor<double>(U.data(), shape.size(), shape.data());
          },
          basix::docstring::FiniteElement__pull_back.c_str())
      .def(
          "apply_dof_transformation",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span((double *)data.data(), data.shape(0));
            self.apply_dof_transformation(data_span, block_size, cell_info);
          },
          basix::docstring::FiniteElement__apply_dof_transformation.c_str())
      .def(
          "apply_dof_transformation_to_transpose",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span((double *)data.data(), data.shape(0));
            self.apply_dof_transformation_to_transpose(data_span, block_size,
                                                       cell_info);
          },
          basix::docstring::FiniteElement__apply_dof_transformation_to_transpose
              .c_str())
      .def(
          "apply_inverse_transpose_dof_transformation",
          [](const FiniteElement& self,
             nb::tensor<double, nb::shape<nb::any>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span((double *)data.data(), data.shape(0));
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
            auto [t, shape] = self.base_transformations();
            return nb::tensor<double>((double *)t.data(), shape.size(), shape.data());
          },
          basix::docstring::FiniteElement__base_transformations.c_str())
      .def(
          "entity_transformations",
          [](const FiniteElement& self)
          {
            auto t = self.entity_transformations();
            nb::dict t2;
            for (auto& [key, data] : t)
            {
              t2[cell_type_to_str(key).c_str()]
                  = nb::tensor<double>(data.first.data(), data.second.size(), data.second.data());
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
      .def_property_readonly("num_entity_dofs",
                             [](const FiniteElement& self)
                             {
                               // TODO: remove this function. Information can
                               // retrieved from entity_dofs.
                               auto& edofs = self.entity_dofs();
                               std::vector<std::vector<int>> num_edofs;
                               for (auto& edofs_d : edofs)
                               {
                                 auto& ndofs = num_edofs.emplace_back();
                                 for (auto& edofs : edofs_d)
                                   ndofs.push_back(edofs.size());
                               }
                               return num_edofs;
                             })
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("num_entity_closure_dofs",
                             [](const FiniteElement& self)
                             {
                               // TODO: remove this function. Information can
                               // retrieved from entity_closure_dofs.
                               auto& edofs = self.entity_closure_dofs();
                               std::vector<std::vector<int>> num_edofs;
                               for (auto& edofs_d : edofs)
                               {
                                 auto& ndofs = num_edofs.emplace_back();
                                 for (auto& edofs : edofs_d)
                                   ndofs.push_back(edofs.size());
                               }
                               return num_edofs;
                             })
      .def_property_readonly("entity_closure_dofs",
                             &FiniteElement::entity_closure_dofs)
      .def_property_readonly("value_size",
                             [](const FiniteElement& self)
                             {
                               return std::accumulate(
                                   self.value_shape().begin(),
                                   self.value_shape().end(), 1,
                                   std::multiplies{});
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
      .def_property_readonly("points",
                             [](const FiniteElement& self)
                             {
                               auto& [x, shape] = self.points();
                               return nb::tensor<double>((double *)x.data(), shape.size(), shape.data(),
                                                          nb::cast(self));
                             })
      .def_property_readonly("interpolation_matrix",
                             [](const FiniteElement& self)
                             {
                               auto& [P, shape] = self.interpolation_matrix();
                               return nb::tensor<double>((double *)P.data(), shape.size(), shape.data(),
                                                          nb::cast(self));
                             })
      .def_property_readonly("dual_matrix",
                             [](const FiniteElement& self)
                             {
                               auto& [D, shape] = self.dual_matrix();
                               return nb::tensor<double>((double *)D.data(), shape.size(), shape.data(),
                                                          nb::cast(self));
                             })
      .def_property_readonly("coefficient_matrix",
                             [](const FiniteElement& self)
                             {
                               auto& [P, shape] = self.coefficient_matrix();
                               return nb::tensor<double>((double *)P.data(), shape.size(), shape.data(),
                                                          nb::cast(self));
                             })
      .def_property_readonly("wcoeffs",
                             [](const FiniteElement& self)
                             {
                               auto& [P, shape] = self.wcoeffs();
                               return nb::tensor<double>((double *)P.data(), shape.size(), shape.data(),
                                                          nb::cast(self));
                             })
      .def_property_readonly(
          "M",
          [](const FiniteElement& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 4>>>,
                             4>& _M
                = self.M();
            std::vector<std::vector<nb::tensor<nb::numpy, double>>> M(
                4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _M[i].size(); ++j)
              {
                auto& mat = _M[i][j];
                M[i].push_back(nb::tensor<nb::numpy, double>((double *)mat.first.data(), mat.second.size(), mat.second.data(),
                                                   nb::cast(self)));
              }
            }
            return M;
          })
      .def_property_readonly(
          "x",
          [](const FiniteElement& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 2>>>,
                             4>& _x
                = self.x();
            std::vector<std::vector<nb::tensor<nb::numpy, double>>> x(
                4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _x[i].size(); ++j)
              {
                auto& vec = _x[i][j];
                x[i].push_back(nb::tensor<nb::numpy, double>((double *)vec.first.data(), vec.second.size(), vec.second.data(),
                                                   nb::cast(self)));
              }
            }
            return x;
          })
      .def_property_readonly("has_tensor_product_factorisation",
                             &FiniteElement::has_tensor_product_factorisation)
      .def_property_readonly("interpolation_nderivs",
                             &FiniteElement::interpolation_nderivs);

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
      .value("bernstein", element::lagrange_variant::bernstein)
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
         int interpolation_nderivs, maps::type map_type, bool discontinuous,
         int highest_complete_degree, int highest_degree) -> FiniteElement
      {
        if (x.size() != 4)
          throw std::runtime_error("x has the wrong size");
        if (M.size() != 4)
          throw std::runtime_error("M has the wrong size");

        std::array<std::vector<cmdspan2_t>, 4> _x;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < x[i].size(); ++j)
          {
            if (x[i][j].ndim() != 2)
              throw std::runtime_error("x has the wrong number of dimensions");
            _x[i].emplace_back((double *)x[i][j].data(), x[i][j].shape(0),
                               x[i][j].shape(1));
          }
        }

        std::array<std::vector<cmdspan4_t>, 4> _M;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < M[i].size(); ++j)
          {
            if (M[i][j].ndim() != 4)
              throw std::runtime_error("M has the wrong number of dimensions");
            _M[i].emplace_back((double *)M[i][j].data(), M[i][j].shape(0),
                               M[i][j].shape(1), M[i][j].shape(2),
                               M[i][j].shape(3));
          }
        }

        std::vector<std::size_t> _vs(value_shape.size());
        for (std::size_t i = 0; i < value_shape.size(); ++i)
          _vs[i] = static_cast<std::size_t>(value_shape[i]);

        return basix::create_custom_element(
            cell_type, _vs,
            cmdspan2_t((double *)wcoeffs.data(), wcoeffs.shape(0), wcoeffs.shape(1)), _x,
            _M, interpolation_nderivs, map_type, discontinuous,
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
      {
        auto [out, shape]
            = basix::compute_interpolation_operator(element_from, element_to);
        return nb::tensor<nb::numpy, double>(out.data(), shape.size(), shape.data());
      },
      basix::docstring::compute_interpolation_operator.c_str());

  m.def(
      "tabulate_polynomial_set",
      [](cell::type celltype, int d, int n,
         const nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>>& x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> _x((double *)
            x.data(), x.shape(0), x.shape(1));
        auto [p, shape] = polyset::tabulate(celltype, d, n, _x);
        return nb::tensor<nb::numpy, double>(p.data(), shape.size(), shape.data());
      },
      basix::docstring::tabulate_polynomial_set.c_str());

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, int m)
      {
        auto [pts, w] = quadrature::make_quadrature(rule, celltype, m);
        std::array<std::size_t, 2> shape = {w.size(), pts.size() / w.size()};
        return std::pair(nb::tensor<nb::numpy, double>(pts.data(), 2, shape.data()),
                         nb::tensor<nb::numpy, double>(w.data(), 1, &shape[0]));
      },
      basix::docstring::make_quadrature__rule_celltype_m.c_str());

  m.def(
      "make_quadrature",
      [](cell::type celltype, int m)
      {
        auto [pts, w] = quadrature::make_quadrature(celltype, m);
        std::array<std::size_t, 2> shape = {w.size(), pts.size() / w.size()};
        return std::pair(nb::tensor<nb::numpy, double>(pts.data(), 2, shape.data()),
                         nb::tensor<nb::numpy, double>(w.data(), 1, &shape[0]));
      },
      basix::docstring::make_quadrature__celltype_m.c_str());

  m.def("index", nb::overload_cast<int>(&basix::indexing::idx),
        basix::docstring::index__p.c_str())
      .def("index", nb::overload_cast<int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q.c_str())
      .def("index", nb::overload_cast<int, int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q_r.c_str());
}
