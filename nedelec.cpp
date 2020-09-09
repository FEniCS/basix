
#include "nedelec.h"
#include "quadrature.h"
#include "simplex.h"
#include <Eigen/SVD>
#include <numeric>

Nedelec2D::Nedelec2D(int k) : _degree(k - 1)
{
  // Reference simplex
  ReferenceSimplex simplex(2);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> Pkp1 = simplex.compute_polynomial_set(_degree + 1);

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) / 2;
  std::vector<int> vec_idx(nv);
  std::iota(vec_idx.begin(), vec_idx.end(), 0);

  // PkH subset
  const int ns = _degree + 1;
  std::vector<int> scalar_idx(ns);
  std::iota(scalar_idx.begin(), scalar_idx.end(), (_degree + 1) * _degree / 2);

  auto [Qpts, Qwts] = make_quadrature_triangle_collapsed(2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts(Pkp1.size(), Qpts.rows());
  for (std::size_t j = 0; j < Pkp1.size(); ++j)
    Pkp1_at_Qpts.row(j) = Pkp1[j].tabulate(Qpts);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      PkH_crossx_coeffs_0(ns, Pkp1.size());
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      PkH_crossx_coeffs_1(ns, Pkp1.size());

  for (int i = 0; i < ns; ++i)
    for (std::size_t k = 0; k < Pkp1.size(); ++k)
    {
      auto w = Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose() * Qpts.col(1)
               * Pkp1_at_Qpts.row(k).transpose();
      PkH_crossx_coeffs_0(i, k) = w.sum();

      auto w2 = -Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose()
                * Qpts.col(0) * Pkp1_at_Qpts.row(k).transpose();
      PkH_crossx_coeffs_1(i, k) = w2.sum();
    }

  // Reproducing code from FIAT to get SVD

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2 + ns, Pkp1.size() * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, Pkp1.size(), nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv * 2, 0, ns, Pkp1.size()) = PkH_crossx_coeffs_0;
  wcoeffs.block(nv * 2, Pkp1.size(), ns, Pkp1.size()) = PkH_crossx_coeffs_1;

  Eigen::JacobiSVD svd(wcoeffs, Eigen::ComputeFullV);

  std::cout << "v=" << svd.matrixV() << "\n";
}

//-----------------------------------------------------------------------------
