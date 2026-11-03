// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <array>
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/maps.h>
#include <basix/mdspan.hpp>
#include <basix/polyset.h>
#include <basix/sobolev-spaces.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace basix_wrappers
{
/// mdspan with dynamic extents in all `d` dimensions.
template <typename T, std::size_t d>
using mdspan_t = basix::md::mdspan<T, basix::md::dextents<std::size_t, d>>;

/// Convert a cell type to its Python-facing string name.
inline std::string cell_type_to_str(basix::cell::type type)
{
  static const std::map<basix::cell::type, std::string> type_to_name
      = {{basix::cell::type::point, "point"},
         {basix::cell::type::interval, "interval"},
         {basix::cell::type::triangle, "triangle"},
         {basix::cell::type::tetrahedron, "tetrahedron"},
         {basix::cell::type::quadrilateral, "quadrilateral"},
         {basix::cell::type::pyramid, "pyramid"},
         {basix::cell::type::prism, "prism"},
         {basix::cell::type::hexahedron, "hexahedron"}};
  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");
  return it->second;
}

/// Create a NumPy array that takes ownership of the data in `x`.
template <typename V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  namespace nb = nanobind;
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::move(x));
  return nb::ndarray<typename _V::value_type, nb::numpy>(
      ptr->data(), ndim, shape,
      nb::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

/// Create a NumPy array that takes ownership of the data in `x`.
template <typename V>
auto as_nbarray(V&& x, const std::initializer_list<std::size_t> shape)
{
  return as_nbarray(x, shape.size(), shape.begin());
}

/// Create a one-dimensional NumPy array that takes ownership of the data
/// in `x`.
template <typename V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::move(x), {x.size()});
}

/// Create a NumPy array from a (data, shape) pair, taking ownership of
/// the data.
template <typename V, std::size_t U>
auto as_nbarrayp(std::pair<V, std::array<std::size_t, U>>&& x)
{
  return as_nbarray(std::move(x.first), x.second.size(), x.second.data());
}

/// Call `f` with its template parameter set to `float` or `double`,
/// depending on `dtype` ('f' or 'd'), converting the result to `R`.
template <typename R, typename F>
R dispatch_dtype(char dtype, F&& f)
{
  if (dtype == 'd')
    return f.template operator()<double>();
  else if (dtype == 'f')
    return f.template operator()<float>();
  else
    throw std::runtime_error("Unsupported finite element dtype.");
}

namespace impl
{
/// @brief Value size of data on a physical cell for a given map type.
///
/// This matches the convention used by `FiniteElement::push_forward`
/// and `FiniteElement::pull_back`.
/// @param[in] map_type The map type.
/// @param[in] gdim The geometric dimension of the physical cell.
/// @param[in] reference_value_size The value size of data on the
/// reference cell.
/// @return The physical value size, or 0 if `map_type` is not
/// supported.
constexpr std::size_t physical_value_size(basix::maps::type map_type,
                                          std::size_t gdim,
                                          std::size_t reference_value_size)
{
  switch (map_type)
  {
  case basix::maps::type::identity:
    return reference_value_size;
  case basix::maps::type::covariantPiola:
  case basix::maps::type::contravariantPiola:
    return gdim;
  case basix::maps::type::doubleCovariantPiola:
  case basix::maps::type::doubleContravariantPiola:
    return gdim * gdim;
  default:
    return 0;
  }
}

/// @brief Tabulate basis functions, for calling from a Numba-compiled
/// kernel.
///
/// This function uses no Python API, allows no C++ exception to escape
/// into just-in-time compiled code, and does not require the GIL. Its
/// address is exposed to Python by `declare_float`.
///
/// @param[in] element Address of a `basix::FiniteElement<T>`. The
/// caller is responsible for keeping the element alive for the duration
/// of the call.
/// @param[in] nd The number of derivatives to compute.
/// @param[in] x The points at which to tabulate, with row-major layout
/// and shape `(num_points, tdim)`.
/// @param[in] num_points The number of points.
/// @param[out] basis Buffer for the basis values, with row-major layout
/// and shape `element.tabulate_shape(nd, num_points)`.
/// @return 0 on success, -1 if the call raised an exception.
template <typename T>
int numba_tabulate(const void* element, int nd, const T* x,
                   std::size_t num_points, T* basis) noexcept
{
  try
  {
    const basix::FiniteElement<T>& e
        = *static_cast<const basix::FiniteElement<T>*>(element);
    const std::size_t tdim = basix::cell::topological_dimension(e.cell_type());
    const std::array<std::size_t, 4> shape = e.tabulate_shape(nd, num_points);
    e.tabulate(nd, mdspan_t<const T, 2>(x, num_points, tdim),
               mdspan_t<T, 4>(basis, shape));
    return 0;
  }
  catch (...)
  {
    return -1;
  }
}

/// @brief Push values forward to, or pull them back from, a physical
/// cell, for calling from a Numba-compiled kernel.
///
/// Values are mapped in groups that share a Jacobian. Unlike
/// `FiniteElement::push_forward` and `FiniteElement::pull_back`, this
/// function writes into a caller-provided buffer and performs no heap
/// allocation.
///
/// This function uses no Python API, allows no C++ exception to escape
/// into just-in-time compiled code, and does not require the GIL. Its
/// address is exposed to Python by `declare_float`.
///
/// @param[in] element Address of a `basix::FiniteElement<T>`. The
/// caller is responsible for keeping the element alive for the duration
/// of the call.
/// @param[in] inverse If 0, push `values` forward to the physical cell.
/// Otherwise, pull `values` back to the reference cell.
/// @param[in] num_jacobians The number of groups of points that share a
/// Jacobian.
/// @param[in] num_points The number of points in each group.
/// @param[in] gdim The geometric dimension of the physical cell.
/// @param[in] values The values to map, with row-major layout and shape
/// `(num_jacobians, num_points, vs)`, where `vs` is the physical value
/// size for a pull-back and the reference value size for a
/// push-forward.
/// @param[in] J The Jacobians, with row-major layout and shape
/// `(num_jacobians, gdim, tdim)`.
/// @param[in] detJ The Jacobian determinants, with length
/// `num_jacobians`.
/// @param[in] K The inverses of the Jacobians, with row-major layout
/// and shape `(num_jacobians, tdim, gdim)`.
/// @param[out] out Buffer for the mapped values, with row-major layout
/// and shape `(num_jacobians, num_points, vs)`, where `vs` is the
/// reference value size for a pull-back and the physical value size for
/// a push-forward.
/// @return 0 on success, -1 if the map is not supported or the call
/// raised an exception.
template <typename T>
int numba_map(const void* element, int inverse, std::size_t num_jacobians,
              std::size_t num_points, std::size_t gdim, const T* values,
              const T* J, const T* detJ, const T* K, T* out) noexcept
{
  try
  {
    const basix::FiniteElement<T>& e
        = *static_cast<const basix::FiniteElement<T>*>(element);
    const std::size_t tdim = basix::cell::topological_dimension(e.cell_type());
    const std::size_t rvs
        = std::accumulate(e.value_shape().begin(), e.value_shape().end(),
                          std::size_t{1}, std::multiplies{});
    const std::size_t pvs = physical_value_size(e.map_type(), gdim, rvs);
    if (pvs == 0)
      return -1;

    using in_t = mdspan_t<const T, 2>;
    using out_t = mdspan_t<T, 2>;
    const std::size_t in_vs = inverse ? pvs : rvs;
    const std::size_t out_vs = inverse ? rvs : pvs;

    // map_fn returns a std::function holding a captureless lambda, so
    // building it here (rather than per Jacobian) keeps the loop free of
    // allocation.
    auto map = e.template map_fn<out_t, in_t, in_t, in_t>();
    for (std::size_t i = 0; i < num_jacobians; ++i)
    {
      out_t _out(out + i * num_points * out_vs, num_points, out_vs);
      in_t _in(values + i * num_points * in_vs, num_points, in_vs);
      in_t _J(J + i * gdim * tdim, gdim, tdim);
      in_t _K(K + i * tdim * gdim, tdim, gdim);
      if (inverse)
        map(_out, _in, _K, 1 / detJ[i], _J);
      else
        map(_out, _in, _J, detJ[i], _K);
    }

    return 0;
  }
  catch (...)
  {
    return -1;
  }
}
} // namespace impl

/// Wrap the scalar-type dependent parts of the interface, i.e.
/// `FiniteElement` and the functions that create or act on one.
/// @param m The nanobind module.
/// @param type String representation of the scalar type (e.g.
/// "float32", "float64").
template <typename T>
void declare_float(nanobind::module_& m, const std::string& type)
{
  namespace nb = nanobind;
  using namespace nb::literals;
  using basix::FiniteElement;

  // Addresses of the native entry points that basix.numba_helpers
  // registers with LLVM for use inside Numba-compiled kernels.
  m.def(("_numba_tabulate_address_" + type).c_str(), []()
        { return reinterpret_cast<std::uintptr_t>(&impl::numba_tabulate<T>); });
  m.def(("_numba_map_address_" + type).c_str(),
        []() { return reinterpret_cast<std::uintptr_t>(&impl::numba_map<T>); });

  std::string name = "FiniteElement_" + type;
  nb::class_<FiniteElement<T>>(m, name.c_str())
      .def_prop_ro("_address", [](const FiniteElement<T>& self)
                   { return reinterpret_cast<std::uintptr_t>(&self); })
      .def(
          "tabulate",
          [](const FiniteElement<T>& self, int n,
             const nb::ndarray<const T, nb::ndim<2>, nb::c_contig>& x)
          {
            mdspan_t<const T, 2> _x(x.data(), x.shape(0), x.shape(1));
            return as_nbarrayp(self.tabulate(n, _x));
          },
          "n"_a, "x"_a)
      .def(
          "tabulate_shape",
          [](const FiniteElement<T>& self, std::size_t nd,
             std::size_t num_points)
          {
            std::array<std::size_t, 4> shape
                = self.tabulate_shape(nd, num_points);
            return nb::make_tuple(shape[0], shape[1], shape[2], shape[3]);
          },
          "nd"_a, "num_points"_a)
      .def("__eq__", &FiniteElement<T>::operator==,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def("hash", &FiniteElement<T>::hash)
      .def(
          "permute_subentity_closure",
          [](const FiniteElement<T>& self,
             const nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig>& d,
             std::uint32_t entity_info, basix::cell::type entity_type)
          {
            std::span<std::int32_t> _d(d.data(), d.shape(0));
            self.permute_subentity_closure(_d, entity_info, entity_type);
          },
          "d"_a.noconvert(), "entity_info"_a, "entity_type"_a)
      .def(
          "permute_subentity_closure",
          [](const FiniteElement<T>& self,
             const nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig>& d,
             std::uint32_t cell_info, basix::cell::type entity_type,
             int entity_index)
          {
            std::span<std::int32_t> _d(d.data(), d.shape(0));
            self.permute_subentity_closure(_d, cell_info, entity_type,
                                           entity_index);
          },
          "d"_a.noconvert(), "cell_info"_a, "entity_type"_a, "entity_index"_a)
      .def(
          "permute_subentity_closure_inv",
          [](const FiniteElement<T>& self,
             const nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig>& d,
             std::uint32_t entity_info, basix::cell::type entity_type)
          {
            std::span<std::int32_t> _d(d.data(), d.shape(0));
            self.permute_subentity_closure_inv(_d, entity_info, entity_type);
          },
          "d"_a.noconvert(), "entity_info"_a, "entity_type"_a)
      .def(
          "permute_subentity_closure_inv",
          [](const FiniteElement<T>& self,
             const nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig>& d,
             std::uint32_t cell_info, basix::cell::type entity_type,
             int entity_index)
          {
            std::span<std::int32_t> _d(d.data(), d.shape(0));
            self.permute_subentity_closure_inv(_d, cell_info, entity_type,
                                               entity_index);
          },
          "d"_a.noconvert(), "cell_info"_a, "entity_type"_a, "entity_index"_a)
      .def(
          "push_forward",
          [](const FiniteElement<T>& self,
             const nb::ndarray<const T, nb::ndim<3>, nb::c_contig>& U,
             const nb::ndarray<const T, nb::ndim<3>, nb::c_contig>& J,
             const nb::ndarray<const T, nb::ndim<1>, nb::c_contig>& detJ,
             const nb::ndarray<const T, nb::ndim<3>, nb::c_contig>& K)
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
          },
          "U"_a, "J"_a, "detJ"_a, "K"_a)
      .def(
          "pull_back",
          [](const FiniteElement<T>& self,
             const nb::ndarray<const T, nb::ndim<3>, nb::c_contig>& u,
             const nb::ndarray<const T, nb::ndim<3>, nb::c_contig>& J,
             const nb::ndarray<const T, nb::ndim<1>, nb::c_contig>& detJ,
             const nb::ndarray<const T, nb::ndim<3>, nb::c_contig>& K)
          {
            auto U
                = self.pull_back(mdspan_t<const T, 3>(u.data(), u.shape(0),
                                                      u.shape(1), u.shape(2)),
                                 mdspan_t<const T, 3>(J.data(), J.shape(0),
                                                      J.shape(1), J.shape(2)),
                                 std::span<const T>(detJ.data(), detJ.shape(0)),
                                 mdspan_t<const T, 3>(K.data(), K.shape(0),
                                                      K.shape(1), K.shape(2)));
            return as_nbarrayp(std::move(U));
          },
          "u"_a, "J"_a, "detJ"_a, "K"_a)
      .def(
          "T_apply",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          { self.T_apply(std::span(u.data(), u.size()), n, cell_info); },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "Tinv_apply",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          { self.Tinv_apply(std::span(u.data(), u.size()), n, cell_info); },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "Tt_apply",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          { self.Tt_apply(std::span(u.data(), u.size()), n, cell_info); },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "Tt_inv_apply",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          { self.Tt_inv_apply(std::span(u.data(), u.size()), n, cell_info); },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "T_apply_right",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          { self.T_apply_right(std::span(u.data(), u.size()), n, cell_info); },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "Tinv_apply_right",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          {
            self.Tinv_apply_right(std::span(u.data(), u.size()), n, cell_info);
          },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "Tt_apply_right",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          { self.Tt_apply_right(std::span(u.data(), u.size()), n, cell_info); },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def(
          "Tt_inv_apply_right",
          [](const FiniteElement<T>& self,
             const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& u, int n,
             std::uint32_t cell_info)
          {
            self.Tt_inv_apply_right(std::span(u.data(), u.size()), n,
                                    cell_info);
          },
          "u"_a.noconvert(), "n"_a, "cell_info"_a)
      .def("base_transformations", [](const FiniteElement<T>& self)
           { return as_nbarrayp(self.base_transformations()); })
      .def("entity_transformations",
           [](const FiniteElement<T>& self)
           {
             // entity_transformations() now returns a const reference to
             // internal storage (previously a by-value copy of the whole
             // map), so each matrix must be copied here rather than moved.
             nb::dict t;
             for (const auto& [key, data] : self.entity_transformations())
             {
               t[cell_type_to_str(key).c_str()] = as_nbarrayp(
                   std::pair<std::vector<T>, std::array<std::size_t, 3>>(data));
             }
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
                       for (auto& dofs : edofs_d)
                         ndofs.push_back(dofs.size());
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
                       for (auto& dofs : edofs_d)
                         ndofs.push_back(dofs.size());
                     }
                     return num_edofs;
                   })
      .def_prop_ro("entity_closure_dofs",
                   &FiniteElement<T>::entity_closure_dofs)
      .def_prop_ro("value_size",
                   [](const FiniteElement<T>& self)
                   {
                     return std::accumulate(self.value_shape().begin(),
                                            self.value_shape().end(),
                                            std::size_t{1}, std::multiplies{});
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
                             4>& _M = self.M();
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
                             4>& _x = self.x();
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
  std::string custom_name = "create_custom_element_" + type;
  m.def(
      custom_name.c_str(),
      [](basix::cell::type cell_type,
         const std::vector<std::size_t>& value_shape,
         const nb::ndarray<const T, nb::ndim<2>, nb::c_contig>& wcoeffs,
         std::vector<
             std::vector<nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>>
             x,
         std::vector<
             std::vector<nb::ndarray<const T, nb::ndim<4>, nb::c_contig>>>
             M,
         int interpolation_nderivs, basix::maps::type map_type,
         basix::sobolev::space sobolev_space, bool discontinuous,
         int embedded_subdegree, int embedded_superdegree,
         basix::polyset::type poly_type) -> FiniteElement<T>
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

        std::array<std::vector<basix::impl::mdspan_t<const T, 4>>, 4> _M;
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
  m.def(
      "compute_interpolation_operator",
      [](const FiniteElement<T>& element_from,
         const FiniteElement<T>& element_to)
      {
        return as_nbarrayp(
            basix::compute_interpolation_operator(element_from, element_to));
      },
      "e0"_a, "e1"_a);

  m.def(("tabulate_polynomial_set_" + type).c_str(),
        [](basix::cell::type celltype, basix::polyset::type polytype, int d,
           int n, const nb::ndarray<const T, nb::ndim<2>, nb::c_contig>& x)
        {
          mdspan_t<const T, 2> _x(x.data(), x.shape(0), x.shape(1));
          return as_nbarrayp(
              basix::polyset::tabulate(celltype, polytype, d, n, _x));
        },
        "celltype"_a, "ptype"_a, "degree"_a, "nderiv"_a, "pts"_a.noconvert());
}

} // namespace basix_wrappers
