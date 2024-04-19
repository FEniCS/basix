// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <basix/cell.h>
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
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

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
void declare_float(nb::module_& m, std::string type)
{
  std::string name = "FiniteElement_" + type;
  nb::class_<FiniteElement<T>>(m, name.c_str())
      .def("tabulate",
           [](const FiniteElement<T>& self, int n,
              nb::ndarray<const T, nb::ndim<2>, nb::c_contig> x)
           {
             mdspan_t<const T, 2> _x(x.data(), x.shape(0), x.shape(1));
             return as_nbarrayp(self.tabulate(n, _x));
           })
      .def("__eq__", &FiniteElement<T>::operator==)
      .def("__hash__", &FiniteElement<T>::hash)
      .def("push_forward",
           [](const FiniteElement<T>& self,
              nb::ndarray<const T, nb::ndim<3>, nb::c_contig> U,
              nb::ndarray<const T, nb::ndim<3>, nb::c_contig> J,
              nb::ndarray<const T, nb::ndim<1>, nb::c_contig> detJ,
              nb::ndarray<const T, nb::ndim<3>, nb::c_contig> K)
           {
             auto u = self.push_forward(
                 mdspan_t<const T, 3>(U.data(), U.shape(0), U.shape(1),
                                      U.shape(2)),
                 mdspan_t<const T, 3>(J.data(), J.shape(0), J.shape(1),
                                      J.shape(2)),
                 std::span<const T>(detJ.data(), detJ.shape(0)),
                 mdspan_t<const T, 3>(K.data(), K.shape(0), K.shape(1),
                                      K.shape(2)));
             return as_nbarrayp(std::move(u));
           })
      .def("pull_back",
           [](const FiniteElement<T>& self,
              nb::ndarray<const T, nb::ndim<3>, nb::c_contig> u,
              nb::ndarray<const T, nb::ndim<3>, nb::c_contig> J,
              nb::ndarray<const T, nb::ndim<1>, nb::c_contig> detJ,
              nb::ndarray<const T, nb::ndim<3>, nb::c_contig> K)
           {
             auto U = self.pull_back(
                 mdspan_t<const T, 3>(u.data(), u.shape(0), u.shape(1),
                                      u.shape(2)),
                 mdspan_t<const T, 3>(J.data(), J.shape(0), J.shape(1),
                                      J.shape(2)),
                 std::span<const T>(detJ.data(), detJ.shape(0)),
                 mdspan_t<const T, 3>(K.data(), K.shape(0), K.shape(1),
                                      K.shape(2)));
             return as_nbarrayp(std::move(U));
           })
      .def("T_apply", [](const FiniteElement<T>& self,
                         nb::ndarray<T, nb::ndim<1>, nb::c_contig> u, int n,
                         std::uint32_t cell_info)
           { self.T_apply(std::span(u.data(), u.size()), n, cell_info); })
      .def("Tt_apply_right",
           [](const FiniteElement<T>& self,
              nb::ndarray<T, nb::ndim<1>, nb::c_contig> u, int n,
              std::uint32_t cell_info) {
             self.Tt_apply_right(std::span(u.data(), u.size()), n,
                                cell_info);
           })
      .def("Tt_inv_apply",
           [](const FiniteElement<T>& self,
              nb::ndarray<T, nb::ndim<1>, nb::c_contig> u, int n,
              std::uint32_t cell_info) {
             self.Tt_inv_apply(std::span(u.data(), u.size()), n,
                               cell_info);
           })
      .def("base_transformations", [](const FiniteElement<T>& self)
           { return as_nbarrayp(self.base_transformations()); })
      .def("entity_transformations",
           [](const FiniteElement<T>& self)
           {
             nb::dict t;
             for (auto& [key, data] : self.entity_transformations())
               t[cell_type_to_str(key).c_str()] = as_nbarrayp(std::move(data));
             return t;
           })
      .def("get_tensor_product_representation", [](const FiniteElement<T>& self)
           { return self.get_tensor_product_representation(); })
      .def_prop_ro("degree", &FiniteElement<T>::degree)
      .def_prop_ro("embedded_superdegree",
                   &FiniteElement<T>::embedded_superdegree)
      .def_prop_ro("embedded_subdegree", &FiniteElement<T>::embedded_subdegree)
      .def_prop_ro("cell_type", &FiniteElement<T>::cell_type)
      .def_prop_ro("polyset_type", &FiniteElement<T>::polyset_type)
      .def_prop_ro("dim", &FiniteElement<T>::dim)
      .def_prop_ro("num_entity_dofs",
                   [](const FiniteElement<T>& self)
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
      .def_prop_ro("entity_dofs", &FiniteElement<T>::entity_dofs)
      .def_prop_ro("num_entity_closure_dofs",
                   [](const FiniteElement<T>& self)
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
                   &FiniteElement<T>::entity_closure_dofs)
      .def_prop_ro("value_size",
                   [](const FiniteElement<T>& self)
                   {
                     return std::accumulate(self.value_shape().begin(),
                                            self.value_shape().end(), 1,
                                            std::multiplies{});
                   })
      .def_prop_ro("value_shape", &FiniteElement<T>::value_shape)
      .def_prop_ro("discontinuous", &FiniteElement<T>::discontinuous)
      .def_prop_ro("family", &FiniteElement<T>::family)
      .def_prop_ro("lagrange_variant", &FiniteElement<T>::lagrange_variant)
      .def_prop_ro("dpc_variant", &FiniteElement<T>::dpc_variant)
      .def_prop_ro("dof_transformations_are_permutations",
                   &FiniteElement<T>::dof_transformations_are_permutations)
      .def_prop_ro("dof_transformations_are_identity",
                   &FiniteElement<T>::dof_transformations_are_identity)
      .def_prop_ro("interpolation_is_identity",
                   &FiniteElement<T>::interpolation_is_identity)
      .def_prop_ro("map_type", &FiniteElement<T>::map_type)
      .def_prop_ro("sobolev_space", &FiniteElement<T>::sobolev_space)
      .def_prop_ro(
          "points",
          [](const FiniteElement<T>& self)
          {
            auto& [x, shape] = self.points();
            return nb::ndarray<const T, nb::ndim<2>, nb::numpy>(
                x.data(), shape.size(), shape.data(), nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "interpolation_matrix",
          [](const FiniteElement<T>& self)
          {
            auto& [P, shape] = self.interpolation_matrix();
            return nb::ndarray<const T, nb::ndim<2>, nb::numpy>(
                P.data(), shape.size(), shape.data(), nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "dual_matrix",
          [](const FiniteElement<T>& self)
          {
            auto& [D, shape] = self.dual_matrix();
            return nb::ndarray<const T, nb::ndim<2>, nb::numpy>(
                D.data(), shape.size(), shape.data(), nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "coefficient_matrix",
          [](const FiniteElement<T>& self)
          {
            auto& [P, shape] = self.coefficient_matrix();
            return nb::ndarray<const T, nb::ndim<2>, nb::numpy>(
                P.data(), shape.size(), shape.data(), nb::handle());
          },
          nb::rv_policy::reference_internal, "Coefficient matrix.")
      .def_prop_ro(
          "wcoeffs",
          [](const FiniteElement<T>& self)
          {
            auto& [w, shape] = self.wcoeffs();
            return nb::ndarray<const T, nb::ndim<2>, nb::numpy>(
                w.data(), shape.size(), shape.data(), nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "M",
          [](const FiniteElement<T>& self)
          {
            const std::array<std::vector<std::pair<std::vector<T>,
                                                   std::array<std::size_t, 4>>>,
                             4>& _M
                = self.M();
            std::vector<std::vector<nb::ndarray<const T, nb::numpy>>> M(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _M[i].size(); ++j)
              {
                auto& mat = _M[i][j];
                M[i].emplace_back(mat.first.data(), mat.second.size(),
                                  mat.second.data(), nb::handle());
              }
            }
            return M;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "x",
          [](const FiniteElement<T>& self)
          {
            const std::array<std::vector<std::pair<std::vector<T>,
                                                   std::array<std::size_t, 2>>>,
                             4>& _x
                = self.x();
            std::vector<std::vector<nb::ndarray<const T, nb::numpy>>> x(4);
            for (int i = 0; i < 4; ++i)
            {
              for (std::size_t j = 0; j < _x[i].size(); ++j)
              {
                auto& vec = _x[i][j];
                x[i].emplace_back(vec.first.data(), vec.second.size(),
                                  vec.second.data(), nb::handle());
              }
            }
            return x;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("has_tensor_product_factorisation",
                   &FiniteElement<T>::has_tensor_product_factorisation)
      .def_prop_ro("interpolation_nderivs",
                   &FiniteElement<T>::interpolation_nderivs)
      .def_prop_ro("dof_ordering", &FiniteElement<T>::dof_ordering)
      .def_prop_ro("dtype",
                   [](const FiniteElement<T>&) -> char
                   {
                     static_assert(std::is_same_v<T, float>
                                   or std::is_same_v<T, double>);
                     if constexpr (std::is_same_v<T, float>)
                       return 'f';
                     else if constexpr (std::is_same_v<T, double>)
                       return 'd';
                   });

  // Create FiniteElement
  m.def(
      "create_custom_element",
      [](cell::type cell_type, const std::vector<std::size_t>& value_shape,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> wcoeffs,
         std::vector<
             std::vector<nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>>
             x,
         std::vector<
             std::vector<nb::ndarray<const T, nb::ndim<4>, nb::c_contig>>>
             M,
         int interpolation_nderivs, maps::type map_type,
         sobolev::space sobolev_space, bool discontinuous,
         int embedded_subdegree, int embedded_superdegree,
         polyset::type poly_type) -> FiniteElement<T>
      {
        if (x.size() != 4)
          throw std::runtime_error("x has the wrong size");
        if (M.size() != 4)
          throw std::runtime_error("M has the wrong size");

        std::array<std::vector<mdspan_t<const T, 2>>, 4> _x;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < x[i].size(); ++j)
          {
            _x[i].emplace_back(x[i][j].data(), x[i][j].shape(0),
                               x[i][j].shape(1));
          }
        }

        std::array<std::vector<impl::mdspan_t<const T, 4>>, 4> _M;
        for (int i = 0; i < 4; ++i)
        {
          for (std::size_t j = 0; j < M[i].size(); ++j)
          {
            _M[i].emplace_back(M[i][j].data(), M[i][j].shape(0),
                               M[i][j].shape(1), M[i][j].shape(2),
                               M[i][j].shape(3));
          }
        }

        return basix::create_custom_element<T>(
            cell_type, value_shape,
            mdspan_t<const T, 2>(wcoeffs.data(), wcoeffs.shape(0),
                                 wcoeffs.shape(1)),
            _x, _M, interpolation_nderivs, map_type, sobolev_space,
            discontinuous, embedded_subdegree, embedded_superdegree, poly_type);
      },
      "cell_type"_a, "value_shape"_a, "wcoeffs"_a.noconvert(),
      "x"_a.noconvert(), "M"_a.noconvert(), "interpolation_nderivs"_a,
      "map_type"_a, "sobolev_space"_a, "discontinuous"_a,
      "embedded_subdegree"_a, "embedded_superdegree"_a, "poly_type"_a);

  // Interpolate between elements
  m.def("compute_interpolation_operator",
        [](const FiniteElement<T>& element_from,
           const FiniteElement<T>& element_to)
        {
          return as_nbarrayp(
              basix::compute_interpolation_operator(element_from, element_to));
        });

  m.def(
      "tabulate_polynomial_set",
      [](cell::type celltype, polyset::type polytype, int d, int n,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> x)
      {
        mdspan_t<const T, 2> _x(x.data(), x.shape(0), x.shape(1));
        return as_nbarrayp(polyset::tabulate(celltype, polytype, d, n, _x));
      },
      "celltype"_a, "polytype"_a, "d"_a, "n"_a, "x"_a.noconvert());
}

} // namespace

NB_MODULE(_basixcpp, m)
{
  m.doc() = "Interface to the Basix C++ library.";
  m.attr("__version__") = basix::version();

  m.def("topology", &cell::topology);
  m.def("geometry", [](cell::type celltype)
        { return as_nbarrayp(cell::geometry<double>(celltype)); });
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity);
  m.def("sub_entity_geometry",
        [](cell::type celltype, int dim, int index)
        {
          return as_nbarrayp(
              cell::sub_entity_geometry<double>(celltype, dim, index));
        });

  m.def("sobolev_space_intersection", &sobolev::space_intersection);

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

  m.def("tabulate_polynomials",
        [](polynomials::type polytype, cell::type celltype, int d,
           nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x)
        {
          mdspan_t<const double, 2> _x(x.data(), x.shape(0), x.shape(1));
          return as_nbarrayp(polynomials::tabulate(polytype, celltype, d, _x));
        });
  m.def("polynomials_dim", &polynomials::dim);
  m.def("create_lattice",
        [](cell::type celltype, int n, lattice::type type, bool exterior,
           lattice::simplex_method method)
        {
          return as_nbarrayp(
              lattice::create<double>(celltype, n, type, exterior, method));
        });

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

  m.def("cell_volume", [](cell::type cell_type) -> double
        { return cell::volume<double>(cell_type); });
  m.def("cell_facet_normals", [](cell::type cell_type)
        { return as_nbarrayp(cell::facet_normals<double>(cell_type)); });
  m.def("cell_facet_reference_volumes",
        [](cell::type cell_type) {
          return as_nbarray(cell::facet_reference_volumes<double>(cell_type));
        });
  m.def("cell_facet_outward_normals",
        [](cell::type cell_type) {
          return as_nbarrayp(cell::facet_outward_normals<double>(cell_type));
        });
  m.def("cell_facet_orientations",
        [](cell::type cell_type)
        {
          std::vector<bool> c = cell::facet_orientations(cell_type);
          std::vector<std::uint8_t> c8(c.begin(), c.end());
          return c8;
        });
  m.def("cell_facet_jacobians", [](cell::type cell_type)
        { return as_nbarrayp(cell::facet_jacobians<double>(cell_type)); });

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

  m.def("create_element",
        [](element::family family_name, cell::type cell, int degree,
           element::lagrange_variant lagrange_variant,
           element::dpc_variant dpc_variant, bool discontinuous,
           const std::vector<int>& dof_ordering, char dtype)
            -> std::variant<FiniteElement<float>, FiniteElement<double>>
        {
          if (dtype == 'd')
          {
            return basix::create_element<double>(family_name, cell, degree,
                                                 lagrange_variant, dpc_variant,
                                                 discontinuous, dof_ordering);
          }
          else if (dtype == 'f')
          {
            return basix::create_element<float>(family_name, cell, degree,
                                                lagrange_variant, dpc_variant,
                                                discontinuous, dof_ordering);
          }
          else
            throw std::runtime_error("Unsupported finite element dtype.");
        });

  m.def("create_tp_element",
        [](element::family family_name, cell::type cell, int degree,
           element::lagrange_variant lagrange_variant,
           element::dpc_variant dpc_variant, bool discontinuous, char dtype)
            -> std::variant<FiniteElement<float>, FiniteElement<double>>
        {
          if (dtype == 'd')
          {
            return basix::create_tp_element<double>(family_name, cell, degree,
                                                    lagrange_variant,
                                                    dpc_variant, discontinuous);
          }
          else if (dtype == 'f')
          {
            return basix::create_tp_element<float>(family_name, cell, degree,
                                                   lagrange_variant,
                                                   dpc_variant, discontinuous);
          }
          else
            throw std::runtime_error("Unsupported finite element dtype.");
        });

  m.def("tp_factors",
        [](element::family family_name, cell::type cell, int degree,
           element::lagrange_variant lagrange_variant,
           element::dpc_variant dpc_variant, bool discontinuous,
           std::vector<int> dof_ordering, char dtype)
            -> std::variant<std::vector<std::vector<FiniteElement<float>>>,
                            std::vector<std::vector<FiniteElement<double>>>>
        {
          if (dtype == 'd')
          {
            return basix::tp_factors<double>(family_name, cell, degree,
                                             lagrange_variant, dpc_variant,
                                             discontinuous, dof_ordering);
          }
          else if (dtype == 'f')
          {
            return basix::tp_factors<float>(family_name, cell, degree,
                                            lagrange_variant, dpc_variant,
                                            discontinuous, dof_ordering);
          }
          else
            throw std::runtime_error("Unsupported finite element dtype.");
        });

  m.def("tp_dof_ordering",
        [](element::family family_name, cell::type cell, int degree,
           element::lagrange_variant lagrange_variant,
           element::dpc_variant dpc_variant,
           bool discontinuous) -> std::vector<int>
        {
          return basix::tp_dof_ordering(family_name, cell, degree,
                                        lagrange_variant, dpc_variant,
                                        discontinuous);
        });

  nb::enum_<polyset::type>(m, "PolysetType")
      .value("standard", polyset::type::standard)
      .value("macroedge", polyset::type::macroedge)
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });

  m.def("superset",
        [](cell::type cell, polyset::type type1, polyset::type type2)
        { return polyset::superset(cell, type1, type2); });

  m.def("restriction",
        [](polyset::type ptype, cell::type cell, cell::type restriction_cell)
        { return polyset::restriction(ptype, cell, restriction_cell); });

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
      });

  m.def("index", nb::overload_cast<int>(&basix::indexing::idx));
  m.def("index", nb::overload_cast<int, int>(&basix::indexing::idx));
  m.def("index", nb::overload_cast<int, int, int>(&basix::indexing::idx));

  declare_float<float>(m, "float32");
  declare_float<double>(m, "float64");
}
