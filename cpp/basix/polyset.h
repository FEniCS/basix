// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "mdspan.hpp"
#include <array>
#include <concepts>
#include <utility>
#include <vector>

/// @brief Polynomial expansion sets.
///
/// In Basix, the bases of the polynomial sets spanned by finite
/// elements are represented by a set of coefficients of orthonormal
/// polynomials on the cell. By orthogonality, we mean that
///
/// \f[\int_R p_ip_j=\begin{cases}1&i=j\\0&i\not=j\end{cases}\f]
///
/// ### Interval
/// Legendre polynomials form a set of orthogonal polynomials on an
/// interval. Legrendre polynomials are a special case of the <a
/// href="https://en.wikipedia.org/wiki/Jacobi_polynomials">Jacobi
/// polynomials</a>, \f$P_n^{\alpha, \beta}\f$, with the weights
/// \f$(\alpha, \beta) = (0, 0)\f$. Using the Jacobi recurrence
/// relation, we have
///
/// \f[ n P^{0,0}_n(x) = (2n - 1) x P^{0,0}_{n - 1}(x) - (n-1)
/// P^{0,0}_{n-2}(x),\f]
///
/// starting with \f$P^{0,0}_0 = 1\f$. Legendre polynomials are
/// orthogonal when integrated over the interval [-1, 1].
///
/// In Basix, the interval [0, 1] is used as the reference, so we apply
/// the transformation \f$x\mapsto 2x-1\f$ to the standard Legendre
/// polynomials. The orthogonal polynomials we use are therefore given
/// by the recurrence relation
///
/// \f[n p_n(x)=(2n-1)(2x-1)p_{n-1}(x)-(n-1)p_{n-2}(x).\f]
///
/// For a given set of points, the recurrence relation can be used to
/// compute the polynomial set up to arbitrary order.
///
/// ### Triangle (Dubiner's basis)
/// See <a href="https://doi.org/10.1016/0045-7825(94)00745-9">Sherwin &
/// Karniadakis (1995)</a>.
///
/// An orthogonal set over a triangle can be obtained with a change of
/// variable to \f$\zeta = 2\frac{1+x}{1 - y} - 1\f$, and a modified
/// polynomial. The polynomial basis becomes:
///
/// \f[ Q_{p,q} = P^{0,0}_p(\zeta) \left(\frac{1-y}{2}\right)^p\ P_q^{2p+1,
/// 0}(y), \f]
///
/// with a Legendre Polynomial of \f$\zeta\f$ and a weighted Jacobi
/// Polynomial of \f$y\f$. In order to calculate these without actually
/// multiplying together the two polynomials directly, we can first
/// compute \f$Q_{p, 0}\f$ using a recurrence relation (since \f$P_0^{2p
/// + 1, 0} = 1\f$):
///
/// \f[ p Q_{p,0} = (2p-1)(2x + y+1)Q_{p-1,0} - (p-1)(1-y)^2 Q_{p-2, 0}.\f]
///
/// Subsequently, we can calculate \f$Q_{p,q}\f$ by building up from
/// \f$Q_{p,0}\f$ with another recurrence relation:
///
/// \f[ Q_{p,q} = (a_0 y + a_1) Q_{p, q-1} + a_2 Q_{p, q-2}, \f]
///
/// where \f$a_0, a_1, a_2\f$ are the coefficients in the Jacobi
/// recurrence relation with \f$(\alpha,\beta) = (2p+1, 0)\f$, see <a
/// href="https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations">Wikipedia</a>.
/// Note that \f$a_2 = 0\f$ when \f$q < 2\f$.
///
/// ### Tetrahedron
/// See <a href="https://doi.org/10.1016/0045-7825(94)00745-9">Sherwin &
/// Karniadakis (1995)</a>.
///
/// Let \f$\zeta = 2\frac{1+x}{y+z} + 1\f$ and \f$\xi = 2\frac{1+y}{1-z}
/// - 1\f$. Orthogonal polynomials on a tetrahedron are then given by
///
/// \f[ Q_{p, q, r} = P^{0,0}_p(\zeta)\left(\frac{y+z}{2}\right)^p\
/// P_q^{2p+1,0}(\xi)\left(\frac{1-z}{2}\right)^q\ P_r^{2(p+q+1), 0}(z).\f]
///
/// This can similarly be built up by first computing \f$Q_{p,0,0}\f$,
/// then \f$Q_{p, q, 0}\f$ and finally \f$Q_{p, q, r}\f$ with recurrence
/// relations on each axis.
///
/// ### Quadrilateral and hexahedron
/// The polynomial sets for quadrilateral and hexahedral elements can be
/// formed by applying the recurrence relation for intervals in each
/// direction.
///
/// ### Prism
/// The polynomial set for a prism can be computed by using using the
/// recurrence relation for the triangle to get the polynomials that are
/// constant with respect to \f$z\f$, then applying the recurrence
/// relation for the interval to compute further polynomials with
/// \f$z\f$s.
///
/// ### Pyramid
/// Orthogonal polynomials on the pyramid element are best calculated in
/// the same way as the tetrahedron, using recurrence relations on each
/// axis. Let \f$\zeta_x = 2\frac{1+x}{1-z} - 1\f$, \f$\zeta_y =
/// 2\frac{1+y}{1-z} - 1\f$. The recurrence relation is then
///
/// \f[Q_{p, q, r} = P^{0,0}_p(\zeta_x) P^{0,0}_q(\zeta_y)
/// \left(\frac{1-z}{2}\right)^{(p+q)} P_r^{2(p+q+1), 0}(z).\f]
///
/// ### Normalisation
/// For each cell type, we obtain an orthonormal set of polynomials by
/// scaling each of the orthogonal polynomials.
///
/// ### Derivatives
/// Recurrence relations can also be used to find the derivatives of the
/// polynomials at given points. For example, on the interval, the first
/// derivatives are given by
///
/// \f[ n P'_n(x) = (2n - 1) \left(P_{n-1}(x) + x P'_{n - 1}(x)\right) +
/// (n-1)P'_{n-2}(x). \f]
///
/// More generally, the \f$k\f$-th derivative is given by
///
/// \f[ n P^k_n(x) = (2n - 1) \left(k P^{k-1}_{n-1}(x) + x P^k_{n -
/// 1}(x)\right)+ (n-1) P^k_{n-2}(x) \f]
///
/// This is now a recurrence relation in both \f$n\f$ and \f$k\f$.
/// Similar recurrence relations can be obtained for the derivatives of
/// all the polynomial sets on the other shapes. Care must be taken with
/// quadratic terms, and cross-terms in two and three dimensions.
namespace basix::polyset
{

/// @brief Cell type
enum class type
{
  standard = 0,
  macroedge = 1,
};

/// @brief Tabulate the orthonormal polynomial basis, and derivatives,
/// at points on the reference cell.
///
/// All derivatives up to the given order are computed. If derivatives
/// are not required, use `n = 0`. For example, order `n = 2` for a 2D
/// cell, will compute the basis \f$\psi, d\psi/dx, d\psi/dy, d^2
/// \psi/dx^2, d^2\psi/dxdy, d^2\psi/dy^2\f$ in that order (0, 0), (1,
/// 0), (0, 1), (2, 0), (1, 1), (0 ,2).
///
/// For an interval cell there are `nderiv + 1` derivatives, for a 2D
/// cell, there are `(nderiv + 1)(nderiv + 2)/2` derivatives, and in 3D,
/// there are `(nderiv + 1)(nderiv + 2)(nderiv + 3)/6`. The ordering is
/// 'triangular' with the lower derivatives appearing first.
///
/// @param[in] celltype Cell type
/// @param[in] ptype The polynomial type
/// @param[in] d Polynomial degree
/// @param[in] n Maximum derivative order. Use n = 0 for the basis only.
/// @param[in] x Points at which to evaluate the basis. The shape is
/// (number of points, geometric dimension).
/// @return Polynomial sets, for each derivative, tabulated at points.
/// The shape is `(number of derivatives computed, number of points,
/// basis index)`.
///
/// - The first index is the derivative. The first entry is the basis
/// itself. Derivatives are stored in triangular (2D) or tetrahedral
/// (3D) ordering, eg if `(p, q)` denotes `p` order derivative with
/// respect to `x` and `q` order derivative with respect to `y`, [0] ->
/// (0, 0), [1] -> (1, 0), [2] -> (0, 1), [3] -> (2, 0), [4] -> (1, 1),
/// [5] -> (0, 2), [6] -> (3, 0),...
/// The function basix::indexing::idx maps tuples `(p, q, r)` to the
/// array index.
///
/// - The second index is the point, with index `i` corresponding to the
/// point in row `i` of @p x.
///
/// - The third index is the basis function index.
/// @todo Does the order for the third index need to be documented?
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 3>>
tabulate(cell::type celltype, polyset::type ptype, int d, int n,
         MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
             const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
             x);

/// @brief Tabulate the orthonormal polynomial basis, and derivatives,
/// at points on the reference cell.
///
/// All derivatives up to the given order are computed. If derivatives
/// are not required, use `n = 0`. For example, order `n = 2` for a 2D
/// cell, will compute the basis \f$\psi, d\psi/dx, d\psi/dy, d^2
/// \psi/dx^2, d^2\psi/dxdy, d^2\psi/dy^2\f$ in that order (0, 0), (1,
/// 0), (0, 1), (2, 0), (1, 1), (0 ,2).
///
/// For an interval cell there are `nderiv + 1` derivatives, for a 2D
/// cell, there are `(nderiv + 1)(nderiv + 2)/2` derivatives, and in 3D,
/// there are `(nderiv + 1)(nderiv + 2)(nderiv + 3)/6`. The ordering is
/// 'triangular' with the lower derivatives appearing first.
///
/// @note This function will be called at runtime when tabulating a
/// finite element, so its performance is critical.
///
/// @param[in,out] P Polynomial sets, for each derivative, tabulated at
/// points. The shape is `(number of derivatives computed, number of
/// points, basis index)`.
///
/// - The first index is the derivative. The first entry is the basis
/// itself. Derivatives are stored in triangular (2D) or tetrahedral
/// (3D) ordering, eg if `(p, q)` denotes `p` order derivative with
/// respect to `x` and `q` order derivative with respect to `y`, [0] ->
/// (0, 0), [1] -> (1, 0), [2] -> (0, 1), [3] -> (2, 0), [4] -> (1, 1),
/// [5] -> (0, 2), [6] -> (3, 0),...
/// The function basix::indexing::idx maps tuples `(p, q, r)` to the array
/// index.
///
/// - The second index is the point, with index `i` corresponding to the
/// point in row `i` of `x`.
///
/// - The third index is the basis function index.
/// @todo Does the order for the third index need to be documented?
/// @param[in] celltype Cell type
/// @param[in] ptype The polynomial type
/// @param[in] d Polynomial degree
/// @param[in] n Maximum derivative order. Use `n=0` for the basis only.
/// @param[in] x Points at which to evaluate the basis. The shape is
/// `(number of points, geometric dimension)`.
template <std::floating_point T>
void tabulate(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    cell::type celltype, polyset::type ptype, int d, int n,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x);

/// @brief Dimension of a polynomial space
/// @param[in] cell Cell type
/// @param[in] ptype Polynomial type
/// @param[in] d Polynomial degree
/// @return The number of terms in the basis spanning a space of
/// polynomial degree `d`.
int dim(cell::type cell, polyset::type ptype, int d);

/// @brief Number of derivatives that the orthonormal basis will have on
/// the given cell.
/// @param[in] cell Cell type
/// @param[in] d Highest derivative order
/// @return Number of derivatives
int nderivs(cell::type cell, int d);

/// @brief Get the polyset types that is a superset of two types on the
/// given cell.
/// @param[in] cell Cell type
/// @param[in] type1 First polyset type
/// @param[in] type2 Decond polyset type
/// @return Superset type
polyset::type superset(cell::type cell, polyset::type type1,
                       polyset::type type2);

/// @brief Get the polyset type that represents the restrictions of a
/// type on a subentity.
/// @param[in] ptype Polyset type
/// @param[in] cell Cell type
/// @param[in] restriction_cell Cell type of the subentity
/// @return Restricted polyset type
polyset::type restriction(polyset::type ptype, cell::type cell,
                          cell::type restriction_cell);

} // namespace basix::polyset
