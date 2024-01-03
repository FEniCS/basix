// Copyright (c) 2020-2023 Chris Richardson, Matthew Scroggs and Garth N. Wells
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
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace basix;

template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

namespace
{
std::string cell_type_to_str(cell::type type)
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

template <typename V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::move(x));
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      ptr->data(), ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

template <typename V>
auto as_nbarray(V&& x, const std::initializer_list<std::size_t> shape)
{
  return as_nbarray(x, shape.size(), shape.begin());
}

template <typename V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::move(x), {x.size()});
}

template <typename V, std::size_t U>
auto as_nbarrayp(std::pair<V, std::array<std::size_t, U>>&& x)
{
  return as_nbarray(std::move(x.first), x.second.size(), x.second.data());
}

template <typename T>
void declare_float(nb::module_& m, std::string /*type*/)
{
  nb::class_<FiniteElement<double>>(m, "FiniteElement")
      .def(
          "tabulate",
          [](const FiniteElement<double>& self, int n,
             nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x)
          {
            mdspan_t<const double, 2> _x(x.data(), x.shape(0), x.shape(1));
            return as_nbarrayp(self.tabulate(n, _x));
          },
          basix::docstring::FiniteElement__tabulate.c_str())
      .def("__eq__", &FiniteElement<double>::operator==)
      .def(
          "push_forward",
          [](const FiniteElement<double>& self,
             nb::ndarray<const double, nb::ndim<3>, nb::c_contig> U,
             nb::ndarray<const double, nb::ndim<3>, nb::c_contig> J,
             nb::ndarray<const double, nb::ndim<1>, nb::c_contig> detJ,
             nb::ndarray<const double, nb::ndim<3>, nb::c_contig> K)
          {
            auto u = self.push_forward(
                mdspan_t<const double, 3>(U.data(), U.shape(0), U.shape(1),
                                          U.shape(2)),
                mdspan_t<const double, 3>(J.data(), J.shape(0), J.shape(1),
                                          J.shape(2)),
                std::span<const double>(detJ.data(), detJ.shape(0)),
                mdspan_t<const double, 3>(K.data(), K.shape(0), K.shape(1),
                                          K.shape(2)));
            return as_nbarrayp(std::move(u));
          },
          basix::docstring::FiniteElement__push_forward.c_str())
      .def(
          "pull_back",
          [](const FiniteElement<double>& self,
             nb::ndarray<const double, nb::ndim<3>, nb::c_contig> u,
             nb::ndarray<const double, nb::ndim<3>, nb::c_contig> J,
             nb::ndarray<const double, nb::ndim<1>, nb::c_contig> detJ,
             nb::ndarray<const double, nb::ndim<3>, nb::c_contig> K)
          {
            auto U = self.pull_back(
                mdspan_t<const double, 3>(u.data(), u.shape(0), u.shape(1),
                                          u.shape(2)),
                mdspan_t<const double, 3>(J.data(), J.shape(0), J.shape(1),
                                          J.shape(2)),
                std::span<const double>(detJ.data(), detJ.shape(0)),
                mdspan_t<const double, 3>(K.data(), K.shape(0), K.shape(1),
                                          K.shape(2)));
            return as_nbarrayp(std::move(U));
          },
          basix::docstring::FiniteElement__pull_back.c_str())
      .def(
          "pre_apply_dof_transformation",
          [](const FiniteElement<double>& self,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            self.pre_apply_dof_transformation(
                std::span(data.data(), data.size()), block_size, cell_info);
          },
          basix::docstring::FiniteElement__pre_apply_dof_transformation.c_str())
      .def(
          "post_apply_transpose_dof_transformation",
          [](const FiniteElement<double>& self,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            self.post_apply_transpose_dof_transformation(
                std::span(data.data(), data.size()), block_size, cell_info);
          },
          basix::docstring::
              FiniteElement__post_apply_transpose_dof_transformation.c_str())
      .def(
          "pre_apply_inverse_transpose_dof_transformation",
          [](const FiniteElement<double>& self,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> data,
             int block_size, std::uint32_t cell_info)
          {
            self.pre_apply_inverse_transpose_dof_transformation(
                std::span(data.data(), data.size()), block_size, cell_info);
          },
          basix::docstring::
              FiniteElement__pre_apply_inverse_transpose_dof_transformation
                  .c_str())
      .def(
          "base_transformations",
          [](const FiniteElement<double>& self)
          { return as_nbarrayp(self.base_transformations()); },
          basix::docstring::FiniteElement__base_transformations.c_str())
      .def(
          "entity_transformations",
          [](const FiniteElement<double>& self)
          {
            nb::dict t;
            for (auto& [key, data] : self.entity_transformations())
              t[cell_type_to_str(key).c_str()] = as_nbarrayp(std::move(data));
            return t;
          },
          basix::docstring::FiniteElement__entity_transformations.c_str())
      .def(
          "get_tensor_product_representation",
          [](const FiniteElement<double>& self)
          { return self.get_tensor_product_representation(); },
          basix::docstring::FiniteElement__get_tensor_product_representation
              .c_str())
      .def_prop_ro("degree", &FiniteElement<double>::degree)
      .def_prop_ro("embedded_superdegree",
                   &FiniteElement<double>::embedded_superdegree)
      .def_prop_ro("embedded_subdegree",
                   &FiniteElement<double>::embedded_subdegree)
      .def_prop_ro("cell_type", &FiniteElement<double>::cell_type)
      .def_prop_ro("polyset_type", &FiniteElement<double>::polyset_type)
      .def_prop_ro("dim", &FiniteElement<double>::dim)
      .def_prop_ro("num_entity_dofs",
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
      .def_prop_ro("entity_dofs", &FiniteElement<double>::entity_dofs)
      .def_prop_ro("num_entity_closure_dofs",
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
      .def_prop_ro("entity_closure_dofs",
                   &FiniteElement<double>::entity_closure_dofs)
      .def_prop_ro("value_size",
                   [](const FiniteElement<double>& self)
                   {
                     return std::accumulate(self.value_shape().begin(),
                                            self.value_shape().end(), 1,
                                            std::multiplies{});
                   })
      .def_prop_ro("value_shape", &FiniteElement<double>::value_shape)
      .def_prop_ro("discontinuous", &FiniteElement<double>::discontinuous)
      .def_prop_ro("family", &FiniteElement<double>::family)
      .def_prop_ro("lagrange_variant", &FiniteElement<double>::lagrange_variant)
      .def_prop_ro("dpc_variant", &FiniteElement<double>::dpc_variant)
      .def_prop_ro("dof_transformations_are_permutations",
                   &FiniteElement<double>::dof_transformations_are_permutations)
      .def_prop_ro("dof_transformations_are_identity",
                   &FiniteElement<double>::dof_transformations_are_identity)
      .def_prop_ro("interpolation_is_identity",
                   &FiniteElement<double>::interpolation_is_identity)
      .def_prop_ro("map_type", &FiniteElement<double>::map_type)
      .def_prop_ro("sobolev_space", &FiniteElement<double>::sobolev_space)
      .def_prop_ro(
          "points",
          [](const FiniteElement<double>& self)
          {
            auto& [x, shape] = self.points();
            return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
                x.data(), shape.size(), shape.data());
          },
          nb::rv_policy::reference_internal, "TODO")
      .def_prop_ro(
          "interpolation_matrix",
          [](const FiniteElement<double>& self)
          {
            auto& [P, shape] = self.interpolation_matrix();
            return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
                P.data(), shape.size(), shape.data());
          },
          nb::rv_policy::reference_internal, "TODO")
      .def_prop_ro(
          "dual_matrix",
          [](const FiniteElement<double>& self)
          {
            auto& [D, shape] = self.dual_matrix();
            return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
                D.data(), shape.size(), shape.data());
          },
          nb::rv_policy::reference_internal, "TODO")
      .def_prop_ro(
          "coefficient_matrix",
          [](const FiniteElement<double>& self)
          {
            auto& [P, shape] = self.coefficient_matrix();
            return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
                P.data(), shape.size(), shape.data());
          },
          nb::rv_policy::reference_internal, "Coefficient matrix.")
      .def_prop_ro(
          "wcoeffs",
          [](const FiniteElement<double>& self)
          {
            auto& [w, shape] = self.wcoeffs();
            return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
                w.data(), shape.size(), shape.data());
          },
          nb::rv_policy::reference_internal, "TODO")
      .def_prop_ro(
          "M",
          [](const FiniteElement<double>& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 4>>>,
                             4>& _M
                = self.M();
            std::vector<std::vector<nb::ndarray<const double, nb::numpy>>> M(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _M[i].size(); ++j)
              {
                auto& mat = _M[i][j];
                M[i].emplace_back(mat.first.data(), mat.second.size(),
                                  mat.second.data());
              }
            }
            return M;
          },
          nb::rv_policy::reference_internal, "TODO")
      .def_prop_ro(
          "x",
          [](const FiniteElement<double>& self)
          {
            const std::array<std::vector<std::pair<std::vector<double>,
                                                   std::array<std::size_t, 2>>>,
                             4>& _x
                = self.x();
            std::vector<std::vector<nb::ndarray<const double, nb::numpy>>> x(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _x[i].size(); ++j)
              {
                auto& vec = _x[i][j];
                x[i].emplace_back(vec.first.data(), vec.second.size(),
                                  vec.second.data());
              }
            }
            return x;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("has_tensor_product_factorisation",
                   &FiniteElement<double>::has_tensor_product_factorisation,
                   "TODO")
      .def_prop_ro("interpolation_nderivs",
                   &FiniteElement<double>::interpolation_nderivs, "TODO")
      .def_prop_ro("dof_ordering", &FiniteElement<double>::dof_ordering,
                   "TODO");
}

} // namespace

NB_MODULE(_basixcpp, m)
{
  m.doc() = "Interface to the Basix C++ library.";
  m.attr("__version__") = basix::version();

  m.def("topology", &cell::topology, basix::docstring::topology.c_str());
  m.def(
      "geometry",
      [](cell::type celltype)
      { return as_nbarrayp(cell::geometry<double>(celltype)); },
      basix::docstring::geometry.c_str());
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity,
        basix::docstring::sub_entity_connectivity.c_str());
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index)
      {
        return as_nbarrayp(
            cell::sub_entity_geometry<double>(celltype, dim, index));
      },
      basix::docstring::sub_entity_geometry.c_str());

  m.def("sobolev_space_intersection", &sobolev::space_intersection,
        basix::docstring::space_intersection.c_str());

  nb::enum_<lattice::type>(m, "LatticeType")
      .value("equispaced", lattice::type::equispaced)
      .value("gll", lattice::type::gll)
      .value("chebyshev", lattice::type::chebyshev)
      .value("gl", lattice::type::gl)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });
  nb::enum_<lattice::simplex_method>(m, "LatticeSimplexMethod")
      .value("none", lattice::simplex_method::none)
      .value("warp", lattice::simplex_method::warp)
      .value("isaac", lattice::simplex_method::isaac)
      .value("centroid", lattice::simplex_method::centroid)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  nb::enum_<polynomials::type>(m, "PolynomialType")
      .value("legendre", polynomials::type::legendre)
      .value("bernstein", polynomials::type::bernstein)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  m.def(
      "tabulate_polynomials",
      [](polynomials::type polytype, cell::type celltype, int d,
         nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x)
      {
        mdspan_t<const double, 2> _x(x.data(), x.shape(0), x.shape(1));
        return as_nbarrayp(polynomials::tabulate(polytype, celltype, d, _x));
      },
      basix::docstring::tabulate_polynomials.c_str());
  m.def("polynomials_dim", &polynomials::dim,
        basix::docstring::polynomials_dim.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior)
      {
        return as_nbarrayp(lattice::create<double>(
            celltype, n, type, exterior, lattice::simplex_method::none));
      },
      basix::docstring::create_lattice__celltype_n_type_exterior.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method)
      {
        return as_nbarrayp(
            lattice::create<double>(celltype, n, type, exterior, method));
      },
      basix::docstring::create_lattice__celltype_n_type_exterior_method
          .c_str());

  nb::enum_<maps::type>(m, "MapType")
      .value("identity", maps::type::identity)
      .value("L2Piola", maps::type::L2Piola)
      .value("covariantPiola", maps::type::covariantPiola)
      .value("contravariantPiola", maps::type::contravariantPiola)
      .value("doubleCovariantPiola", maps::type::doubleCovariantPiola)
      .value("doubleContravariantPiola", maps::type::doubleContravariantPiola)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  nb::enum_<sobolev::space>(m, "SobolevSpace")
      .value("L2", sobolev::space::L2)
      .value("H1", sobolev::space::H1)
      .value("H2", sobolev::space::H2)
      .value("H3", sobolev::space::H3)
      .value("HInf", sobolev::space::HInf)
      .value("HDiv", sobolev::space::HDiv)
      .value("HCurl", sobolev::space::HCurl)
      .value("HEin", sobolev::space::HEin)
      .value("HDivDiv", sobolev::space::HDivDiv)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  nb::enum_<quadrature::type>(m, "QuadratureType")
      .value("Default", quadrature::type::Default)
      .value("gauss_jacobi", quadrature::type::gauss_jacobi)
      .value("gll", quadrature::type::gll)
      .value("xiao_gimbutas", quadrature::type::xiao_gimbutas)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  nb::enum_<cell::type>(m, "CellType", nb::is_arithmetic())
      .value("point", cell::type::point)
      .value("interval", cell::type::interval)
      .value("triangle", cell::type::triangle)
      .value("tetrahedron", cell::type::tetrahedron)
      .value("quadrilateral", cell::type::quadrilateral)
      .value("hexahedron", cell::type::hexahedron)
      .value("prism", cell::type::prism)
      .value("pyramid", cell::type::pyramid)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  m.def(
      "cell_volume",
      [](cell::type cell_type) -> double
      { return cell::volume<double>(cell_type); },
      basix::docstring::cell_volume.c_str());
  m.def(
      "cell_facet_normals",
      [](cell::type cell_type)
      { return as_nbarrayp(cell::facet_normals<double>(cell_type)); },
      basix::docstring::cell_facet_normals.c_str());
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type)
      { return as_nbarray(cell::facet_reference_volumes<double>(cell_type)); },
      basix::docstring::cell_facet_reference_volumes.c_str());
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type)
      { return as_nbarrayp(cell::facet_outward_normals<double>(cell_type)); },
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
      { return as_nbarrayp(cell::facet_jacobians<double>(cell_type)); },
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
      .value("Hermite", element::family::Hermite)
      .value("iso", element::family::iso)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  // nb::class_<FiniteElement<double>>(m, "FiniteElement")
  //     .def(
  //         "tabulate",
  //         [](const FiniteElement<double>& self, int n,
  //            nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x)
  //         {
  //           mdspan_t<const double, 2> _x(x.data(), x.shape(0), x.shape(1));
  //           return as_nbarrayp(self.tabulate(n, _x));
  //         },
  //         basix::docstring::FiniteElement__tabulate.c_str())
  //     .def("__eq__", &FiniteElement<double>::operator==)
  //     .def(
  //         "push_forward",
  //         [](const FiniteElement<double>& self,
  //            nb::ndarray<const double, nb::ndim<3>, nb::c_contig> U,
  //            nb::ndarray<const double, nb::ndim<3>, nb::c_contig> J,
  //            nb::ndarray<const double, nb::ndim<1>, nb::c_contig> detJ,
  //            nb::ndarray<const double, nb::ndim<3>, nb::c_contig> K)
  //         {
  //           auto u = self.push_forward(
  //               mdspan_t<const double, 3>(U.data(), U.shape(0), U.shape(1),
  //                                         U.shape(2)),
  //               mdspan_t<const double, 3>(J.data(), J.shape(0), J.shape(1),
  //                                         J.shape(2)),
  //               std::span<const double>(detJ.data(), detJ.shape(0)),
  //               mdspan_t<const double, 3>(K.data(), K.shape(0), K.shape(1),
  //                                         K.shape(2)));
  //           return as_nbarrayp(std::move(u));
  //         },
  //         basix::docstring::FiniteElement__push_forward.c_str())
  //     .def(
  //         "pull_back",
  //         [](const FiniteElement<double>& self,
  //            nb::ndarray<const double, nb::ndim<3>, nb::c_contig> u,
  //            nb::ndarray<const double, nb::ndim<3>, nb::c_contig> J,
  //            nb::ndarray<const double, nb::ndim<1>, nb::c_contig> detJ,
  //            nb::ndarray<const double, nb::ndim<3>, nb::c_contig> K)
  //         {
  //           auto U = self.pull_back(
  //               mdspan_t<const double, 3>(u.data(), u.shape(0), u.shape(1),
  //                                         u.shape(2)),
  //               mdspan_t<const double, 3>(J.data(), J.shape(0), J.shape(1),
  //                                         J.shape(2)),
  //               std::span<const double>(detJ.data(), detJ.shape(0)),
  //               mdspan_t<const double, 3>(K.data(), K.shape(0), K.shape(1),
  //                                         K.shape(2)));
  //           return as_nbarrayp(std::move(U));
  //         },
  //         basix::docstring::FiniteElement__pull_back.c_str())
  //     .def(
  //         "pre_apply_dof_transformation",
  //         [](const FiniteElement<double>& self,
  //            nb::ndarray<double, nb::ndim<1>, nb::c_contig> data,
  //            int block_size, std::uint32_t cell_info)
  //         {
  //           self.pre_apply_dof_transformation(std::span(data.data(),
  //           data.size()),
  //                                         block_size, cell_info);
  //         },
  //         basix::docstring::FiniteElement__pre_apply_dof_transformation.c_str())
  //     .def(
  //         "post_apply_transpose_dof_transformation",
  //         [](const FiniteElement<double>& self,
  //            nb::ndarray<double, nb::ndim<1>, nb::c_contig> data,
  //            int block_size, std::uint32_t cell_info)
  //         {
  //           self.post_apply_transpose_dof_transformation(
  //               std::span(data.data(), data.size()), block_size, cell_info);
  //         },
  //         basix::docstring::FiniteElement__post_apply_transpose_dof_transformation
  //             .c_str())
  //     .def(
  //         "pre_apply_inverse_transpose_dof_transformation",
  //         [](const FiniteElement<double>& self,
  //            nb::ndarray<double, nb::ndim<1>, nb::c_contig> data,
  //            int block_size, std::uint32_t cell_info)
  //         {
  //           self.pre_apply_inverse_transpose_dof_transformation(
  //               std::span(data.data(), data.size()), block_size, cell_info);
  //         },
  //         basix::docstring::
  //             FiniteElement__pre_apply_inverse_transpose_dof_transformation.c_str())
  //     .def(
  //         "base_transformations",
  //         [](const FiniteElement<double>& self)
  //         { return as_nbarrayp(self.base_transformations()); },
  //         basix::docstring::FiniteElement__base_transformations.c_str())
  //     .def(
  //         "entity_transformations",
  //         [](const FiniteElement<double>& self)
  //         {
  //           nb::dict t;
  //           for (auto& [key, data] : self.entity_transformations())
  //             t[cell_type_to_str(key).c_str()] =
  //             as_nbarrayp(std::move(data));
  //           return t;
  //         },
  //         basix::docstring::FiniteElement__entity_transformations.c_str())
  //     .def(
  //         "get_tensor_product_representation",
  //         [](const FiniteElement<double>& self)
  //         { return self.get_tensor_product_representation(); },
  //         basix::docstring::FiniteElement__get_tensor_product_representation
  //             .c_str())
  //     .def_prop_ro("degree", &FiniteElement<double>::degree)
  //     .def_prop_ro("embedded_superdegree",
  //                  &FiniteElement<double>::embedded_superdegree)
  //     .def_prop_ro("embedded_subdegree",
  //                  &FiniteElement<double>::embedded_subdegree)
  //     .def_prop_ro("cell_type", &FiniteElement<double>::cell_type)
  //     .def_prop_ro("polyset_type", &FiniteElement<double>::polyset_type)
  //     .def_prop_ro("dim", &FiniteElement<double>::dim)
  //     .def_prop_ro("num_entity_dofs",
  //                  [](const FiniteElement<double>& self)
  //                  {
  //                    // TODO: remove this function. Information can
  //                    // retrieved from entity_dofs.
  //                    auto& edofs = self.entity_dofs();
  //                    std::vector<std::vector<int>> num_edofs;
  //                    for (auto& edofs_d : edofs)
  //                    {
  //                      auto& ndofs = num_edofs.emplace_back();
  //                      for (auto& edofs : edofs_d)
  //                        ndofs.push_back(edofs.size());
  //                    }
  //                    return num_edofs;
  //                  })
  //     .def_prop_ro("entity_dofs", &FiniteElement<double>::entity_dofs)
  //     .def_prop_ro("num_entity_closure_dofs",
  //                  [](const FiniteElement<double>& self)
  //                  {
  //                    // TODO: remove this function. Information can
  //                    // retrieved from entity_closure_dofs.
  //                    auto& edofs = self.entity_closure_dofs();
  //                    std::vector<std::vector<int>> num_edofs;
  //                    for (auto& edofs_d : edofs)
  //                    {
  //                      auto& ndofs = num_edofs.emplace_back();
  //                      for (auto& edofs : edofs_d)
  //                        ndofs.push_back(edofs.size());
  //                    }
  //                    return num_edofs;
  //                  })
  //     .def_prop_ro("entity_closure_dofs",
  //                  &FiniteElement<double>::entity_closure_dofs)
  //     .def_prop_ro("value_size",
  //                  [](const FiniteElement<double>& self)
  //                  {
  //                    return std::accumulate(self.value_shape().begin(),
  //                                           self.value_shape().end(), 1,
  //                                           std::multiplies{});
  //                  })
  //     .def_prop_ro("value_shape", &FiniteElement<double>::value_shape)
  //     .def_prop_ro("discontinuous", &FiniteElement<double>::discontinuous)
  //     .def_prop_ro("family", &FiniteElement<double>::family)
  //     .def_prop_ro("lagrange_variant",
  //     &FiniteElement<double>::lagrange_variant) .def_prop_ro("dpc_variant",
  //     &FiniteElement<double>::dpc_variant)
  //     .def_prop_ro("dof_transformations_are_permutations",
  //                  &FiniteElement<double>::dof_transformations_are_permutations)
  //     .def_prop_ro("dof_transformations_are_identity",
  //                  &FiniteElement<double>::dof_transformations_are_identity)
  //     .def_prop_ro("interpolation_is_identity",
  //                  &FiniteElement<double>::interpolation_is_identity)
  //     .def_prop_ro("map_type", &FiniteElement<double>::map_type)
  //     .def_prop_ro("sobolev_space", &FiniteElement<double>::sobolev_space)
  //     .def_prop_ro(
  //         "points",
  //         [](const FiniteElement<double>& self)
  //         {
  //           auto& [x, shape] = self.points();
  //           return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
  //               x.data(), shape.size(), shape.data());
  //         },
  //         nb::rv_policy::reference_internal, "TODO")
  //     .def_prop_ro(
  //         "interpolation_matrix",
  //         [](const FiniteElement<double>& self)
  //         {
  //           auto& [P, shape] = self.interpolation_matrix();
  //           return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
  //               P.data(), shape.size(), shape.data());
  //         },
  //         nb::rv_policy::reference_internal, "TODO")
  //     .def_prop_ro(
  //         "dual_matrix",
  //         [](const FiniteElement<double>& self)
  //         {
  //           auto& [D, shape] = self.dual_matrix();
  //           return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
  //               D.data(), shape.size(), shape.data());
  //         },
  //         nb::rv_policy::reference_internal, "TODO")
  //     .def_prop_ro(
  //         "coefficient_matrix",
  //         [](const FiniteElement<double>& self)
  //         {
  //           auto& [P, shape] = self.coefficient_matrix();
  //           return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
  //               P.data(), shape.size(), shape.data());
  //         },
  //         nb::rv_policy::reference_internal, "Coefficient matrix.")
  //     .def_prop_ro(
  //         "wcoeffs",
  //         [](const FiniteElement<double>& self)
  //         {
  //           auto& [w, shape] = self.wcoeffs();
  //           return nb::ndarray<const double, nb::ndim<2>, nb::numpy>(
  //               w.data(), shape.size(), shape.data());
  //         },
  //         nb::rv_policy::reference_internal, "TODO")
  //     .def_prop_ro(
  //         "M",
  //         [](const FiniteElement<double>& self)
  //         {
  //           const std::array<std::vector<std::pair<std::vector<double>,
  //                                                  std::array<std::size_t,
  //                                                  4>>>,
  //                            4>& _M
  //               = self.M();
  //           std::vector<std::vector<nb::ndarray<const double, nb::numpy>>>
  //           M(4); for (int i = 0; i < 4; ++i)
  //           {
  //             for (std::size_t j = 0; j < _M[i].size(); ++j)
  //             {
  //               auto& mat = _M[i][j];
  //               M[i].emplace_back(mat.first.data(), mat.second.size(),
  //                                 mat.second.data());
  //             }
  //           }
  //           return M;
  //         },
  //         nb::rv_policy::reference_internal, "TODO")
  //     .def_prop_ro(
  //         "x",
  //         [](const FiniteElement<double>& self)
  //         {
  //           const std::array<std::vector<std::pair<std::vector<double>,
  //                                                  std::array<std::size_t,
  //                                                  2>>>,
  //                            4>& _x
  //               = self.x();
  //           std::vector<std::vector<nb::ndarray<const double, nb::numpy>>>
  //           x(4); for (int i = 0; i < 4; ++i)
  //           {
  //             for (std::size_t j = 0; j < _x[i].size(); ++j)
  //             {
  //               auto& vec = _x[i][j];
  //               x[i].emplace_back(vec.first.data(), vec.second.size(),
  //                                 vec.second.data());
  //             }
  //           }
  //           return x;
  //         },
  //         nb::rv_policy::reference_internal)
  //     .def_prop_ro("has_tensor_product_factorisation",
  //                  &FiniteElement<double>::has_tensor_product_factorisation,
  //                  "TODO")
  //     .def_prop_ro("interpolation_nderivs",
  //                  &FiniteElement<double>::interpolation_nderivs, "TODO")
  //     .def_prop_ro("dof_ordering", &FiniteElement<double>::dof_ordering,
  //                  "TODO");

  declare_float<double>(m, "float64");

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
      .value("vtk", element::lagrange_variant::vtk)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  nb::enum_<element::dpc_variant>(m, "DPCVariant")
      .value("unset", element::dpc_variant::unset)
      .value("simplex_equispaced", element::dpc_variant::simplex_equispaced)
      .value("simplex_gll", element::dpc_variant::simplex_gll)
      .value("horizontal_equispaced",
             element::dpc_variant::horizontal_equispaced)
      .value("horizontal_gll", element::dpc_variant::horizontal_gll)
      .value("diagonal_equispaced", element::dpc_variant::diagonal_equispaced)
      .value("diagonal_gll", element::dpc_variant::diagonal_gll)
      .value("legendre", element::dpc_variant::legendre)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  // Create FiniteElement
  m.def(
      "create_custom_element",
      [](cell::type cell_type, const std::vector<std::size_t>& value_shape,
         nb::ndarray<const double, nb::ndim<2>, nb::c_contig> wcoeffs,
         std::vector<
             std::vector<nb::ndarray<const double, nb::ndim<2>, nb::c_contig>>>
             x,
         std::vector<
             std::vector<nb::ndarray<const double, nb::ndim<4>, nb::c_contig>>>
             M,
         int interpolation_nderivs, maps::type map_type,
         sobolev::space sobolev_space, bool discontinuous,
         int embedded_subdegree, int embedded_superdegree,
         polyset::type poly_type) -> FiniteElement<double>
      {
        if (x.size() != 4)
          throw std::runtime_error("x has the wrong size");
        if (M.size() != 4)
          throw std::runtime_error("M has the wrong size");

        std::array<std::vector<mdspan_t<const double, 2>>, 4> _x;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < x[i].size(); ++j)
          {
            _x[i].emplace_back(x[i][j].data(), x[i][j].shape(0),
                               x[i][j].shape(1));
          }
        }

        std::array<std::vector<impl::mdspan_t<const double, 4>>, 4> _M;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < M[i].size(); ++j)
          {
            _M[i].emplace_back(M[i][j].data(), M[i][j].shape(0),
                               M[i][j].shape(1), M[i][j].shape(2),
                               M[i][j].shape(3));
          }
        }

        return basix::create_custom_element<double>(
            cell_type, value_shape,
            mdspan_t<const double, 2>(wcoeffs.data(), wcoeffs.shape(0),
                                      wcoeffs.shape(1)),
            _x, _M, interpolation_nderivs, map_type, sobolev_space,
            discontinuous, embedded_subdegree, embedded_superdegree, poly_type);
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
      nb::arg("family_name"), nb::arg("cell_name"), nb::arg("degree"),
      nb::arg("lagrange_variant") = element::lagrange_variant::unset,
      nb::arg("dpc_variant") = element::dpc_variant::unset,
      nb::arg("discontinuous") = false,
      nb::arg("dof_ordering") = std::vector<int>(),
      basix::docstring::
          create_element__family_cell_degree_lvariant_dvariant_discontinuous_dof_ordering
              .c_str());

  // Interpolate between elements
  m.def(
      "compute_interpolation_operator",
      [](const FiniteElement<double>& element_from,
         const FiniteElement<double>& element_to)
      {
        return as_nbarrayp(
            basix::compute_interpolation_operator(element_from, element_to));
      },
      basix::docstring::compute_interpolation_operator.c_str());

  nb::enum_<polyset::type>(m, "PolysetType")
      .value("standard", polyset::type::standard)
      .value("macroedge", polyset::type::macroedge)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

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
         nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x)
      {
        mdspan_t<const double, 2> _x(x.data(), x.shape(0), x.shape(1));
        return as_nbarrayp(polyset::tabulate(celltype, polytype, d, n, _x));
      },
      basix::docstring::tabulate_polynomial_set.c_str());

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, polyset::type polytype,
         int m)
      {
        auto [pts, w]
            = quadrature::make_quadrature<double>(rule, celltype, polytype, m);
        std::array shape{w.size(), pts.size() / w.size()};
        return std::pair(as_nbarray(std::move(pts), shape.size(), shape.data()),
                         as_nbarray(std::move(w)));
      },
      basix::docstring::make_quadrature__rule_celltype_polytype_m.c_str());

  m.def("index", nb::overload_cast<int>(&basix::indexing::idx),
        basix::docstring::index__p.c_str());
  m.def("index", nb::overload_cast<int, int>(&basix::indexing::idx),
        basix::docstring::index__p_q.c_str());
  m.def("index", nb::overload_cast<int, int, int>(&basix::indexing::idx),
        basix::docstring::index__p_q_r.c_str());
}
