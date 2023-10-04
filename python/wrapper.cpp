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
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>
#include <string>

namespace py = pybind11;
using namespace basix;

namespace stdex = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
template <typename T, std::size_t d>
using mdspan_t
    = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T,
                    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

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

} // namespace

PYBIND11_MODULE(_basixcpp, m)
{
  m.doc() = R"(
Interface to the Basix C++ library.
)";

  m.attr("__version__") = basix::version();

  m.def("topology", &cell::topology, basix::docstring::topology.c_str());
  m.def(
      "geometry",
      [](cell::type celltype)
      {
        auto [x, shape] = cell::geometry<double>(celltype);
        return py::array_t<double>(shape, x.data());
      },
      basix::docstring::geometry.c_str());
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity,
        basix::docstring::sub_entity_connectivity.c_str());
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index)
      {
        auto [x, shape]
            = cell::sub_entity_geometry<double>(celltype, dim, index);
        return py::array_t<double>(shape, x.data());
      },
      basix::docstring::sub_entity_geometry.c_str());

  m.def("sobolev_space_intersection", &sobolev::space_intersection,
        basix::docstring::space_intersection.c_str());

  py::enum_<lattice::type>(m, "LatticeType")
      .value("equispaced", lattice::type::equispaced)
      .value("gll", lattice::type::gll)
      .value("chebyshev", lattice::type::chebyshev)
      .value("gl", lattice::type::gl);

  py::enum_<lattice::simplex_method>(m, "LatticeSimplexMethod")
      .value("none", lattice::simplex_method::none)
      .value("warp", lattice::simplex_method::warp)
      .value("isaac", lattice::simplex_method::isaac)
      .value("centroid", lattice::simplex_method::centroid);

  py::enum_<polynomials::type>(m, "PolynomialType")
      .value("legendre", polynomials::type::legendre)
      .value("bernstein", polynomials::type::bernstein);

  m.def(
      "tabulate_polynomials",
      [](polynomials::type polytype, cell::type celltype, int d,
         const py::array_t<double, py::array::c_style>& x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const double,
                      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
            _x(x.data(), x.shape(0), x.shape(1));
        auto [p, shape] = polynomials::tabulate(polytype, celltype, d, _x);
        return py::array_t<double>(shape, p.data());
      },
      basix::docstring::tabulate_polynomials.c_str());
  m.def("polynomials_dim", &polynomials::dim,
        basix::docstring::polynomials_dim.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior)
      {
        auto [x, shape] = lattice::create<double>(
            celltype, n, type, exterior, lattice::simplex_method::none);
        return py::array_t<double>(shape, x.data());
      },
      basix::docstring::create_lattice__celltype_n_type_exterior.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method)
      {
        auto [x, shape]
            = lattice::create<double>(celltype, n, type, exterior, method);
        return py::array_t<double>(shape, x.data());
      },
      basix::docstring::create_lattice__celltype_n_type_exterior_method
          .c_str());

  py::enum_<maps::type>(m, "MapType")
      .value("identity", maps::type::identity)
      .value("L2Piola", maps::type::L2Piola)
      .value("covariantPiola", maps::type::covariantPiola)
      .value("contravariantPiola", maps::type::contravariantPiola)
      .value("doubleCovariantPiola", maps::type::doubleCovariantPiola)
      .value("doubleContravariantPiola", maps::type::doubleContravariantPiola);

  py::enum_<sobolev::space>(m, "SobolevSpace")
      .value("L2", sobolev::space::L2)
      .value("H1", sobolev::space::H1)
      .value("H2", sobolev::space::H2)
      .value("H3", sobolev::space::H3)
      .value("HInf", sobolev::space::HInf)
      .value("HDiv", sobolev::space::HDiv)
      .value("HCurl", sobolev::space::HCurl)
      .value("HEin", sobolev::space::HEin)
      .value("HDivDiv", sobolev::space::HDivDiv);

  py::enum_<quadrature::type>(m, "QuadratureType")
      .value("Default", quadrature::type::Default)
      .value("gauss_jacobi", quadrature::type::gauss_jacobi)
      .value("gll", quadrature::type::gll)
      .value("xiao_gimbutas", quadrature::type::xiao_gimbutas);

  py::enum_<cell::type>(m, "CellType")
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
      [](cell::type cell_type) -> double
      { return cell::volume<double>(cell_type); },
      basix::docstring::cell_volume.c_str());
  m.def(
      "cell_facet_normals",
      [](cell::type cell_type)
      {
        auto [n, shape] = cell::facet_normals<double>(cell_type);
        return py::array_t<double>(shape, n.data());
      },
      basix::docstring::cell_facet_normals.c_str());
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type)
      {
        std::vector<double> v
            = cell::facet_reference_volumes<double>(cell_type);
        std::array<std::size_t, 1> shape = {v.size()};
        return py::array_t<double>(shape, v.data());
      },
      basix::docstring::cell_facet_reference_volumes.c_str());
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type)
      {
        auto [n, shape] = cell::facet_outward_normals<double>(cell_type);
        return py::array_t<double>(shape, n.data());
      },
      basix::docstring::cell_facet_outward_normals.c_str());
  m.def("cell_facet_orientations", &cell::facet_orientations,
        basix::docstring::cell_facet_orientations.c_str());
  m.def(
      "cell_facet_jacobians",
      [](cell::type cell_type)
      {
        auto [J, shape] = cell::facet_jacobians<double>(cell_type);
        return py::array_t<double>(shape, J.data());
      },
      basix::docstring::cell_facet_jacobians.c_str());

  py::enum_<element::family>(m, "ElementFamily")
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
      .value("Hermite", element::family::Hermite)
      .value("iso", element::family::iso);

  py::class_<FiniteElement<double>>(m, "FiniteElement", "Finite Element")
      .def(
          "tabulate",
          [](const FiniteElement<double>& self, int n,
             const py::array_t<double, py::array::c_style>& x)
          {
            if (x.ndim() != 2)
              throw std::runtime_error("x has the wrong size");
            MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const double,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
                _x(x.data(), x.shape(0), x.shape(1));
            auto [t, shape] = self.tabulate(n, _x);
            return py::array_t<double>(shape, t.data());
          },
          basix::docstring::FiniteElement__tabulate.c_str())
      .def("__eq__", &FiniteElement<double>::operator==)
      .def(
          "push_forward",
          [](const FiniteElement<double>& self,
             const py::array_t<double, py::array::c_style>& U,
             const py::array_t<double, py::array::c_style>& J,
             const py::array_t<double, py::array::c_style>& detJ,
             const py::array_t<double, py::array::c_style>& K)
          {
            auto [u, shape] = self.push_forward(
                mdspan_t<const double, 3>(U.data(), U.shape(0), U.shape(1),
                                          U.shape(2)),
                mdspan_t<const double, 3>(J.data(), J.shape(0), J.shape(1),
                                          J.shape(2)),
                std::span<const double>(detJ.data(), detJ.size()),
                mdspan_t<const double, 3>(K.data(), K.shape(0), K.shape(1),
                                          K.shape(2)));
            return py::array_t<double>(shape, u.data());
          },
          basix::docstring::FiniteElement__push_forward.c_str())
      .def(
          "pull_back",
          [](const FiniteElement<double>& self,
             const py::array_t<double, py::array::c_style>& u,
             const py::array_t<double, py::array::c_style>& J,
             const py::array_t<double, py::array::c_style>& detJ,
             const py::array_t<double, py::array::c_style>& K)
          {
            auto [U, shape] = self.pull_back(
                mdspan_t<const double, 3>(u.data(), u.shape(0), u.shape(1),
                                          u.shape(2)),
                mdspan_t<const double, 3>(J.data(), J.shape(0), J.shape(1),
                                          J.shape(2)),
                std::span<const double>(detJ.data(), detJ.size()),
                mdspan_t<const double, 3>(K.data(), K.shape(0), K.shape(1),
                                          K.shape(2)));
            return py::array_t<double>(shape, U.data());
          },
          basix::docstring::FiniteElement__pull_back.c_str())
      .def(
          "apply_dof_transformation",
          [](const FiniteElement<double>& self, py::array_t<double>& data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span(data.mutable_data(), data.size());
            self.apply_dof_transformation(data_span, block_size, cell_info);
            return py::array_t<double>(data_span.size(), data_span.data());
          },
          basix::docstring::FiniteElement__apply_dof_transformation.c_str())
      .def(
          "apply_dof_transformation_to_transpose",
          [](const FiniteElement<double>& self, py::array_t<double>& data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span(data.mutable_data(), data.size());
            self.apply_dof_transformation_to_transpose(data_span, block_size,
                                                       cell_info);
            return py::array_t<double>(data_span.size(), data_span.data());
          },
          basix::docstring::FiniteElement__apply_dof_transformation_to_transpose
              .c_str())
      .def(
          "apply_inverse_transpose_dof_transformation",
          [](const FiniteElement<double>& self, py::array_t<double>& data,
             int block_size, std::uint32_t cell_info)
          {
            std::span<double> data_span(data.mutable_data(), data.size());
            self.apply_inverse_transpose_dof_transformation(
                data_span, block_size, cell_info);
            return py::array_t<double>(data_span.size(), data_span.data());
          },
          basix::docstring::
              FiniteElement__apply_inverse_transpose_dof_transformation.c_str())
      .def(
          "base_transformations",
          [](const FiniteElement<double>& self)
          {
            auto [t, shape] = self.base_transformations();
            return py::array_t<double>(shape, t.data());
          },
          basix::docstring::FiniteElement__base_transformations.c_str())
      .def(
          "entity_transformations",
          [](const FiniteElement<double>& self)
          {
            auto t = self.entity_transformations();
            py::dict t2;
            for (auto& [key, data] : t)
            {
              t2[cell_type_to_str(key).c_str()]
                  = py::array_t<double>(data.second, data.first.data());
            }
            return t2;
          },
          basix::docstring::FiniteElement__entity_transformations.c_str())
      .def(
          "get_tensor_product_representation",
          [](const FiniteElement<double>& self)
          { return self.get_tensor_product_representation(); },
          basix::docstring::FiniteElement__get_tensor_product_representation
              .c_str())
      .def_property_readonly("degree", &FiniteElement<double>::degree)
      .def_property_readonly("highest_degree",
                             &FiniteElement<double>::highest_degree)
      .def_property_readonly("highest_complete_degree",
                             &FiniteElement<double>::highest_complete_degree)
      .def_property_readonly("cell_type", &FiniteElement<double>::cell_type)
      .def_property_readonly("polyset_type",
                             &FiniteElement<double>::polyset_type)
      .def_property_readonly("dim", &FiniteElement<double>::dim)
      .def_property_readonly("num_entity_dofs",
                             [](const FiniteElement<double>& self)
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
      .def_property_readonly("entity_dofs", &FiniteElement<double>::entity_dofs)
      .def_property_readonly("num_entity_closure_dofs",
                             [](const FiniteElement<double>& self)
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
                             &FiniteElement<double>::entity_closure_dofs)
      .def_property_readonly("value_size",
                             [](const FiniteElement<double>& self)
                             {
                               return std::accumulate(
                                   self.value_shape().begin(),
                                   self.value_shape().end(), 1,
                                   std::multiplies{});
                             })
      .def_property_readonly("value_shape", &FiniteElement<double>::value_shape)
      .def_property_readonly("discontinuous",
                             &FiniteElement<double>::discontinuous)
      .def_property_readonly("family", &FiniteElement<double>::family)
      .def_property_readonly("lagrange_variant",
                             &FiniteElement<double>::lagrange_variant)
      .def_property_readonly("dpc_variant", &FiniteElement<double>::dpc_variant)
      .def_property_readonly(
          "dof_transformations_are_permutations",
          &FiniteElement<double>::dof_transformations_are_permutations)
      .def_property_readonly(
          "dof_transformations_are_identity",
          &FiniteElement<double>::dof_transformations_are_identity)
      .def_property_readonly("interpolation_is_identity",
                             &FiniteElement<double>::interpolation_is_identity)
      .def_property_readonly("map_type", &FiniteElement<double>::map_type)
      .def_property_readonly("sobolev_space",
                             &FiniteElement<double>::sobolev_space)
      .def_property_readonly("points",
                             [](const FiniteElement<double>& self)
                             {
                               auto& [x, shape] = self.points();
                               return py::array_t<double>(shape, x.data(),
                                                          py::cast(self));
                             })
      .def_property_readonly("interpolation_matrix",
                             [](const FiniteElement<double>& self)
                             {
                               auto& [P, shape] = self.interpolation_matrix();
                               return py::array_t<double>(shape, P.data(),
                                                          py::cast(self));
                             })
      .def_property_readonly("dual_matrix",
                             [](const FiniteElement<double>& self)
                             {
                               auto& [D, shape] = self.dual_matrix();
                               return py::array_t<double>(shape, D.data(),
                                                          py::cast(self));
                             })
      .def_property_readonly("coefficient_matrix",
                             [](const FiniteElement<double>& self)
                             {
                               auto& [P, shape] = self.coefficient_matrix();
                               return py::array_t<double>(shape, P.data(),
                                                          py::cast(self));
                             })
      .def_property_readonly("wcoeffs",
                             [](const FiniteElement<double>& self)
                             {
                               auto& [P, shape] = self.wcoeffs();
                               return py::array_t<double>(shape, P.data(),
                                                          py::cast(self));
                             })
      .def_property_readonly(
          "M",
          [](const FiniteElement<double>& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 4>>>,
                             4>& _M
                = self.M();
            std::vector<std::vector<py::array_t<double, py::array::c_style>>> M(
                4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _M[i].size(); ++j)
              {
                auto& mat = _M[i][j];
                M[i].push_back(py::array_t<double>(mat.second, mat.first.data(),
                                                   py::cast(self)));
              }
            }
            return M;
          })
      .def_property_readonly(
          "x",
          [](const FiniteElement<double>& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 2>>>,
                             4>& _x
                = self.x();
            std::vector<std::vector<py::array_t<double, py::array::c_style>>> x(
                4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _x[i].size(); ++j)
              {
                auto& vec = _x[i][j];
                x[i].push_back(py::array_t<double>(vec.second, vec.first.data(),
                                                   py::cast(self)));
              }
            }
            return x;
          })
      .def_property_readonly(
          "has_tensor_product_factorisation",
          &FiniteElement<double>::has_tensor_product_factorisation)
      .def_property_readonly("interpolation_nderivs",
                             &FiniteElement<double>::interpolation_nderivs)
      .def_property_readonly("dof_ordering",
                             &FiniteElement<double>::dof_ordering);

  py::enum_<element::lagrange_variant>(m, "LagrangeVariant")
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

  py::enum_<element::dpc_variant>(m, "DPCVariant")
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
         const py::array_t<double, py::array::c_style>& wcoeffs,
         const std::vector<
             std::vector<py::array_t<double, py::array::c_style>>>& x,
         const std::vector<
             std::vector<py::array_t<double, py::array::c_style>>>& M,
         int interpolation_nderivs, maps::type map_type,
         sobolev::space sobolev_space, bool discontinuous,
         int highest_complete_degree, int highest_degree,
         polyset::type poly_type) -> FiniteElement<double>
      {
        if (x.size() != 4)
          throw std::runtime_error("x has the wrong size");
        if (M.size() != 4)
          throw std::runtime_error("M has the wrong size");

        std::array<std::vector<impl::mdspan_t<const double, 2>>, 4> _x;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < x[i].size(); ++j)
          {
            if (x[i][j].ndim() != 2)
              throw std::runtime_error("x has the wrong number of dimensions");
            _x[i].emplace_back(x[i][j].data(), x[i][j].shape(0),
                               x[i][j].shape(1));
          }
        }

        std::array<std::vector<impl::mdspan_t<const double, 4>>, 4> _M;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < M[i].size(); ++j)
          {
            if (M[i][j].ndim() != 4)
              throw std::runtime_error("M has the wrong number of dimensions");
            _M[i].emplace_back(M[i][j].data(), M[i][j].shape(0),
                               M[i][j].shape(1), M[i][j].shape(2),
                               M[i][j].shape(3));
          }
        }

        std::vector<std::size_t> _vs(value_shape.size());
        for (std::size_t i = 0; i < value_shape.size(); ++i)
          _vs[i] = static_cast<std::size_t>(value_shape[i]);

        return basix::create_custom_element<double>(
            cell_type, _vs,
            mdspan_t<const double, 2>(wcoeffs.data(), wcoeffs.shape(0),
                                      wcoeffs.shape(1)),
            _x, _M, interpolation_nderivs, map_type, sobolev_space,
            discontinuous, highest_complete_degree, highest_degree, poly_type);
      },
      basix::docstring::create_custom_element.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant lvariant, element::dpc_variant dvariant,
         bool discontinuous,
         const std::vector<int>& dof_ordering) -> FiniteElement<double>
      {
        return basix::create_element<double>(family_name, cell_name, degree,
                                             lvariant, dvariant, discontinuous,
                                             dof_ordering);
      },
      py::arg("family_name"), py::arg("cell_name"), py::arg("degree"),
      py::arg("lagrange_variant") = element::lagrange_variant::unset,
      py::arg("dpc_variant") = element::dpc_variant::unset,
      py::arg("discontinuous") = false,
      py::arg("dof_ordering") = std::vector<int>(),
      basix::docstring::
          create_element__family_cell_degree_lvariant_dvariant_discontinuous_dof_ordering
              .c_str());

  // Interpolate between elements
  m.def(
      "compute_interpolation_operator",
      [](const FiniteElement<double>& element_from,
         const FiniteElement<double>& element_to)
          -> const py::array_t<double, py::array::c_style>
      {
        auto [out, shape] = basix::compute_interpolation_operator<double>(
            element_from, element_to);
        return py::array_t<double>(shape, out.data());
      },
      basix::docstring::compute_interpolation_operator.c_str());

  py::enum_<polyset::type>(m, "PolysetType")
      .value("standard", polyset::type::standard)
      .value("macroedge", polyset::type::macroedge);

  m.def(
      "superset",
      [](cell::type cell, polyset::type type1, polyset::type type2)
      { return polyset::superset(cell, type1, type2); },
      basix::docstring::superset.c_str());

  m.def(
      "restriction",
      [](polyset::type ptype, cell::type cell, cell::type restriction_cell)
      { return polyset::restriction(ptype, cell, restriction_cell); },
      basix::docstring::restriction.c_str());

  m.def(
      "tabulate_polynomial_set",
      [](cell::type celltype, polyset::type polytype, int d, int n,
         const py::array_t<double, py::array::c_style>& x)
      {
        if (x.ndim() != 2)
          throw std::runtime_error("x has the wrong number of dimensions");
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const double,
                      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
            _x(x.data(), x.shape(0), x.shape(1));
        auto [p, shape] = polyset::tabulate(celltype, polytype, d, n, _x);
        return py::array_t<double>(shape, p.data());
      },
      basix::docstring::tabulate_polynomial_set.c_str());

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, polyset::type polytype,
         int m)
      {
        auto [pts, w]
            = quadrature::make_quadrature<double>(rule, celltype, polytype, m);
        std::array<std::size_t, 2> shape = {w.size(), pts.size() / w.size()};
        return std::pair(py::array_t<double>(shape, pts.data()),
                         py::array_t<double>(w.size(), w.data()));
      },
      basix::docstring::make_quadrature__rule_celltype_polytype_m.c_str());

  m.def("index", py::overload_cast<int>(&basix::indexing::idx),
        basix::docstring::index__p.c_str())
      .def("index", py::overload_cast<int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q.c_str())
      .def("index", py::overload_cast<int, int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q_r.c_str());
}
