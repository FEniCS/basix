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
#include <basix/sobolev-spaces.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <span>
#include <string>
#include <vector>

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

template <typename V, typename U>
auto as_nbndarray(V& array, U& shape)
{
  std::size_t dim = shape.size();
  auto data = array.data();
  std::unique_ptr<V> x_ptr = std::make_unique<V>(std::move(array));
  auto capsule = nb::capsule(x_ptr.get(), [](void* p) noexcept
                             { std::unique_ptr<V>(reinterpret_cast<V*>(p)); });
  x_ptr.release();
  return nb::ndarray<nb::numpy, typename V::value_type>(
      const_cast<typename V::value_type*>(data), dim, shape.data(), capsule);
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
        auto [x, shape] = cell::geometry(celltype);
        return as_nbndarray(x, shape);
      },
      basix::docstring::geometry.c_str());
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity,
        basix::docstring::sub_entity_connectivity.c_str());
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index)
      {
        auto [x, shape] = cell::sub_entity_geometry(celltype, dim, index);
        return as_nbndarray(x, shape);
      },
      basix::docstring::sub_entity_geometry.c_str());

  m.def("sobolev_space_intersection", &sobolev::space_intersection,
        basix::docstring::space_intersection.c_str());

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
         nb::ndarray<double> x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> _x(
            (double*)x.data(), x.shape(0), x.shape(1));
        auto [p, shape] = polynomials::tabulate(polytype, celltype, d, _x);
        return as_nbndarray(p, shape);
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
        return as_nbndarray(x, shape);
      },
      basix::docstring::create_lattice__celltype_n_type_exterior.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method)
      {
        auto [x, shape] = lattice::create(celltype, n, type, exterior, method);
        return as_nbndarray(x, shape);
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

  nb::enum_<sobolev::space>(m, "SobolevSpace")
      .value("L2", sobolev::space::L2)
      .value("H1", sobolev::space::H1)
      .value("H2", sobolev::space::H2)
      .value("H3", sobolev::space::H3)
      .value("HInf", sobolev::space::HInf)
      .value("HDiv", sobolev::space::HDiv)
      .value("HCurl", sobolev::space::HCurl)
      .value("HEin", sobolev::space::HEin)
      .value("HDivDiv", sobolev::space::HDivDiv);

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
        return as_nbndarray(n, shape);
      },
      basix::docstring::cell_facet_normals.c_str());
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type)
      {
        std::vector<double> v = cell::facet_reference_volumes(cell_type);
        std::array<std::size_t, 1> shape = {v.size()};
        return as_nbndarray(v, shape);
      },
      basix::docstring::cell_facet_reference_volumes.c_str());
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type)
      {
        auto [n, shape] = cell::facet_outward_normals(cell_type);
        return as_nbndarray(n, shape);
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
        return as_nbndarray(J, shape);
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
             const nb::ndarray<nb::numpy, double, nb::shape<nb::any, nb::any>>&
                 x)
          {
            if (x.ndim() != 2)
              throw std::runtime_error("x has the wrong size");
            stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> _x(
                (const double*)x.data(), x.shape(0), x.shape(1));
            auto [t, shape] = self.tabulate(n, _x);
            return as_nbndarray(t, shape);
          },
          basix::docstring::FiniteElement__tabulate.c_str())
      .def("__eq__", &FiniteElement::operator==)
      .def(
          "push_forward",
          [](const FiniteElement& self, const nb::ndarray<double>& U,
             const nb::ndarray<double>& J, const nb::ndarray<double>& detJ,
             const nb::ndarray<double>& K)
          {
            auto [u, shape] = self.push_forward(
                cmdspan3_t(U.data(), U.shape(0), U.shape(1), U.shape(2)),
                cmdspan3_t(J.data(), J.shape(0), J.shape(1), J.shape(2)),
                std::span<const double>(detJ.data(), detJ.shape(0)),
                cmdspan3_t(K.data(), K.shape(0), K.shape(1), K.shape(2)));
            return as_nbndarray(u, shape);
          },
          basix::docstring::FiniteElement__push_forward.c_str())
      .def(
          "pull_back",
          [](const FiniteElement& self, const nb::ndarray<double>& u,
             const nb::ndarray<double>& J, const nb::ndarray<double>& detJ,
             const nb::ndarray<double>& K)
          {
            auto [U, shape] = self.pull_back(
                cmdspan3_t(u.data(), u.shape(0), u.shape(1), u.shape(2)),
                cmdspan3_t(J.data(), J.shape(0), J.shape(1), J.shape(2)),
                std::span<const double>(detJ.data(), detJ.shape(0)),
                cmdspan3_t(K.data(), K.shape(0), K.shape(1), K.shape(2)));
            return as_nbndarray(U, shape);
          },
          basix::docstring::FiniteElement__pull_back.c_str())
      .def(
          "apply_dof_transformation",
          [](const FiniteElement& self, nb::ndarray<double>& data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span(data.data(), data.shape(0));
            self.apply_dof_transformation(data_span, block_size, cell_info);
          },
          basix::docstring::FiniteElement__apply_dof_transformation.c_str())
      .def(
          "apply_dof_transformation_to_transpose",
          [](const FiniteElement& self, nb::ndarray<double>& data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span(data.data(), data.shape(0));
            self.apply_dof_transformation_to_transpose(data_span, block_size,
                                                       cell_info);
          },
          basix::docstring::FiniteElement__apply_dof_transformation_to_transpose
              .c_str())
      .def(
          "apply_inverse_transpose_dof_transformation",
          [](const FiniteElement& self, nb::ndarray<double>& data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span(data.data(), data.shape(0));
            self.apply_inverse_transpose_dof_transformation(
                data_span, block_size, cell_info);
            std::array<std::size_t, 1> size = {data.shape(0)};
            return as_nbndarray(data_span, size);
          },
          basix::docstring::
              FiniteElement__apply_inverse_transpose_dof_transformation.c_str())
      .def(
          "base_transformations",
          [](const FiniteElement& self)
          {
            auto [t, shape] = self.base_transformations();
            return as_nbndarray(t, shape);
          },
          basix::docstring::FiniteElement__base_transformations.c_str())
      .def(
          "entity_transformations",
          [](const FiniteElement& self)
          {
            auto t = self.entity_transformations();
            nb::dict t2;
            for (auto& [key, data] : t)
              t2[cell_type_to_str(key).c_str()]
                  = as_nbndarray(data.first, data.second);
            return t2;
          },
          basix::docstring::FiniteElement__entity_transformations.c_str())
      .def(
          "get_tensor_product_representation",
          [](const FiniteElement& self)
          { return self.get_tensor_product_representation(); },
          basix::docstring::FiniteElement__get_tensor_product_representation
              .c_str())
      .def_prop_ro("degree", &FiniteElement::degree)
      .def_prop_ro("highest_degree", &FiniteElement::highest_degree)
      .def_prop_ro("highest_complete_degree",
                             &FiniteElement::highest_complete_degree)
      .def_prop_ro("cell_type", &FiniteElement::cell_type)
      .def_prop_ro("dim", &FiniteElement::dim)
      .def_prop_ro("num_entity_dofs",
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
      .def_prop_ro("entity_dofs", &FiniteElement::entity_dofs)
      .def_prop_ro("num_entity_closure_dofs",
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
      .def_prop_ro("entity_closure_dofs",
                             &FiniteElement::entity_closure_dofs)
      .def_prop_ro("value_size",
                             [](const FiniteElement& self)
                             {
                               return std::accumulate(
                                   self.value_shape().begin(),
                                   self.value_shape().end(), 1,
                                   std::multiplies{});
                             })
      .def_prop_ro("value_shape", &FiniteElement::value_shape)
      .def_prop_ro("discontinuous", &FiniteElement::discontinuous)
      .def_prop_ro("family", &FiniteElement::family)
      .def_prop_ro("lagrange_variant",
                             &FiniteElement::lagrange_variant)
      .def_prop_ro("dpc_variant", &FiniteElement::dpc_variant)
      .def_prop_ro(
          "dof_transformations_are_permutations",
          &FiniteElement::dof_transformations_are_permutations)
      .def_prop_ro("dof_transformations_are_identity",
                             &FiniteElement::dof_transformations_are_identity)
      .def_prop_ro("interpolation_is_identity",
                             &FiniteElement::interpolation_is_identity)
      .def_prop_ro("map_type", &FiniteElement::map_type)
      .def_prop_ro("sobolev_space", &FiniteElement::sobolev_space)
      .def_prop_ro("points",
                             [](const FiniteElement& self)
                             {
                               auto& [x, shape] = self.points();
                               return as_nbndarray(x, shape);
                             })
      .def_prop_ro("interpolation_matrix",
                             [](const FiniteElement& self)
                             {
                               auto& [P, shape] = self.interpolation_matrix();
                               return as_nbndarray(P, shape);
                             })
      .def_prop_ro("dual_matrix",
                             [](const FiniteElement& self)
                             {
                               auto& [D, shape] = self.dual_matrix();
                               return as_nbndarray(D, shape);
                             })
      .def_prop_ro("coefficient_matrix",
                             [](const FiniteElement& self)
                             {
                               auto& [P, shape] = self.coefficient_matrix();
                               return as_nbndarray(P, shape);
                             })
      .def_prop_ro("wcoeffs",
                             [](const FiniteElement& self)
                             {
                               auto& [P, shape] = self.wcoeffs();
                               return as_nbndarray(P, shape);
                             })
      .def_prop_ro(
          "M",
          [](const FiniteElement& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 4>>>,
                             4>& _M
                = self.M();
            std::vector<std::vector<nb::ndarray<nb::numpy, double>>> M(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _M[i].size(); ++j)
              {
                auto& mat = _M[i][j];
                M[i].push_back(as_nbndarray(mat.first, mat.second));
              }
            }
            return M;
          })
      .def_prop_ro(
          "x",
          [](const FiniteElement& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 2>>>,
                             4>& _x
                = self.x();
            std::vector<std::vector<nb::ndarray<nb::numpy, double>>> x(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _x[i].size(); ++j)
              {
                auto& vec = _x[i][j];
                x[i].push_back(as_nbndarray(vec.first, vec.second));
              }
            }
            return x;
          })
      .def_prop_ro("has_tensor_product_factorisation",
                             &FiniteElement::has_tensor_product_factorisation)
      .def_prop_ro("interpolation_nderivs",
                             &FiniteElement::interpolation_nderivs)
      .def_prop_ro("dof_ordering", &FiniteElement::dof_ordering);

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
         const nb::ndarray<double>& wcoeffs,
         const std::vector<std::vector<nb::ndarray<double>>>& x,
         const std::vector<std::vector<nb::ndarray<double>>>& M,
         int interpolation_nderivs, maps::type map_type,
         sobolev::space sobolev_space, bool discontinuous,
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
            _x[i].emplace_back((double*)x[i][j].data(), x[i][j].shape(0),
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
            _M[i].emplace_back((double*)M[i][j].data(), M[i][j].shape(0),
                               M[i][j].shape(1), M[i][j].shape(2),
                               M[i][j].shape(3));
          }
        }

        std::vector<std::size_t> _vs(value_shape.size());
        for (std::size_t i = 0; i < value_shape.size(); ++i)
          _vs[i] = static_cast<std::size_t>(value_shape[i]);

        return basix::create_custom_element(
            cell_type, _vs,
            cmdspan2_t((double*)wcoeffs.data(), wcoeffs.shape(0),
                       wcoeffs.shape(1)),
            _x, _M, interpolation_nderivs, map_type, sobolev_space,
            discontinuous, highest_complete_degree, highest_degree);
      },
      basix::docstring::create_custom_element.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant lvariant, element::dpc_variant dvariant,
         bool discontinuous,
         const std::vector<int>& dof_ordering) -> FiniteElement
      {
        return basix::create_element(family_name, cell_name, degree, lvariant,
                                     dvariant, discontinuous, dof_ordering);
      },
      nb::arg("family_name"), nb::arg("cell_name"), nb::arg("degree"),
      nb::arg("lagrange_variant") = element::lagrange_variant::unset,
      nb::arg("dpc_variant") = element::dpc_variant::unset,
      nb::arg("discontinuous") = false,
      nb::arg("dof_ordering") = std::vector<int>(),
      basix::docstring::create_element.c_str());

  // Interpolate between elements
  m.def(
      "compute_interpolation_operator",
      [](const FiniteElement& element_from, const FiniteElement& element_to)
      {
        auto [out, shape]
            = basix::compute_interpolation_operator(element_from, element_to);
        return as_nbndarray(out, shape);
      },
      basix::docstring::compute_interpolation_operator.c_str());

  m.def(
      "tabulate_polynomial_set",
      [](cell::type celltype, int d, int n,
         const nb::ndarray<nb::numpy, double, nb::shape<nb::any, nb::any>>& x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        stdex::mdspan<const double, stdex::dextents<std::size_t, 2>> _x(
            (double*)x.data(), x.shape(0), x.shape(1));
        auto [p, shape] = polyset::tabulate(celltype, d, n, _x);
        return as_nbndarray(p, shape);
      },
      basix::docstring::tabulate_polynomial_set.c_str());

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, int m)
      {
        auto [pts, w] = quadrature::make_quadrature(rule, celltype, m);
        const std::array<std::size_t, 1> wshape = {w.size()};
        std::array<std::size_t, 2> shape = {w.size(), pts.size() / w.size()};
        return std::pair(as_nbndarray(pts, shape), as_nbndarray(w, wshape));
      },
      basix::docstring::make_quadrature__rule_celltype_m.c_str());

  m.def(
      "make_quadrature",
      [](cell::type celltype, int m)
      {
        auto [pts, w] = quadrature::make_quadrature(celltype, m);
        const std::array<std::size_t, 1> wshape = {w.size()};
        std::array<std::size_t, 2> shape = {w.size(), pts.size() / w.size()};
        return std::pair(as_nbndarray(pts, shape), as_nbndarray(w, wshape));
      },
      basix::docstring::make_quadrature__celltype_m.c_str());

  m.def("index", nb::overload_cast<int>(&basix::indexing::idx),
        basix::docstring::index__p.c_str())
      .def("index", nb::overload_cast<int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q.c_str())
      .def("index", nb::overload_cast<int, int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q_r.c_str());
}
