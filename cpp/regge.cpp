// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "regge.h"
#include "polynomial-set.h"
#include <iostream>

using namespace libtab;

namespace
{

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_regge_space(Cell::Type celltype, int degree)
{
  const int tdim = Cell::topological_dimension(celltype);
  const int nc = tdim * (tdim + 1) / 2;
  int basis_size;

  if (celltype == Cell::Type::triangle)
    basis_size = (degree + 1) * (degree + 2) / 2;
  else if (celltype == Cell::Type::tetrahedron)
    basis_size = (degree + 1) * (degree + 2) * (degree + 3) / 6;
  else
    throw std::runtime_error("Unsupported celltype");

  int ndofs = basis_size * nc;
  int psize = basis_size * tdim * tdim;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs(
      ndofs, psize);
  wcoeffs.setZero();
  for (int i = 0; i < tdim; ++i)
  {
    std::cout << i << " - " << i * basis_size * (tdim + 1) / 2 << " "
              << basis_size * tdim * i << "\n";

    wcoeffs.block(i * psize / 4, basis_size * tdim * i, basis_size * tdim,
                  basis_size * tdim)
        = Eigen::MatrixXd::Identity(basis_size * tdim, basis_size * tdim);
  }

  std::cout << "\nwoceffs = \n[" << wcoeffs << "]\n";

  return wcoeffs;
}

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_regge_dual(Cell::Type celltype, int degree)
{
  const int tdim = Cell::topological_dimension(celltype);

  int basis_size;

  if (celltype == Cell::Type::triangle)
    basis_size = (degree + 1) * (degree + 2) / 2;
  else if (celltype == Cell::Type::tetrahedron)
    basis_size = (degree + 1) * (degree + 2) * (degree + 3) / 6;
  else
    throw std::runtime_error("Unsupported celltype");

  int ndofs = basis_size * (tdim + 1) * tdim / 2;
  int space_size = basis_size * tdim * tdim;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat(
      ndofs, space_size);
  auto topology = Cell::topology(celltype);
  auto geometry = Cell::geometry(celltype);

  // dof counter
  int dof = 0;
  for (std::size_t dim = 1; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          entity_geom = Cell::sub_entity_geometry(celltype, dim, i);

      Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> point
          = entity_geom.row(0);
      Cell::Type ct = Cell::simplex_type(dim);
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          lattice = Cell::create_lattice(ct, degree + 2, false);
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts(
          lattice.rows(), entity_geom.cols());

      for (int j = 0; j < lattice.rows(); ++j)
      {
        pts.row(j) = entity_geom.row(0);
        for (int k = 0; k < entity_geom.rows() - 1; ++k)
          pts.row(j)
              += (entity_geom.row(k + 1) - entity_geom.row(0)) * lattice(j, k);
      }

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          basis = PolynomialSet::tabulate(celltype, degree, 0, pts)[0];

      std::vector<int>& vert_ids = topology[dim][i];
      int ntangents = dim * (dim + 1) / 2;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          edge_ts(ntangents, geometry.cols());
      int c = 0;
      for (std::size_t s = 0; s < dim; ++s)
        for (std::size_t d = s + 1; d < dim + 1; ++d)
          edge_ts.row(c++)
              = geometry.row(vert_ids[d]) - geometry.row(vert_ids[s]);

      std::cout << "dim = " << dim << " " << i << "\n";
      std::cout << "pts = " << pts << "\n";
      std::cout << "basis = " << basis << "\n";

      std::cout << "tangents = " << edge_ts << "\n";

      for (int j = 0; j < ntangents; ++j)
      {

        // outer product
        Eigen::MatrixXd vvt = edge_ts.row(j).transpose() * edge_ts.row(j);
        Eigen::Map<Eigen::VectorXd> vvt_flat(vvt.data(),
                                             vvt.rows() * vvt.cols());
        for (int k = 0; k < pts.rows(); ++k)
        {
          std::cout << "dof=" << dof << "\n";
          // outer product, outer(outer(t, t), basis)
          Eigen::MatrixXd vvt_b = vvt_flat * basis.row(k);
          dualmat.row(dof++) = Eigen::Map<Eigen::RowVectorXd>(
              vvt_b.data(), vvt_b.rows() * vvt_b.cols());
          std::cout << "vvt * basis = " << vvt_b << "\n";
        }
      }
    }
  }

  return dualmat;
}

}; // namespace

Regge::Regge(Cell::Type celltype, int k) : FiniteElement(celltype, k)
{
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs
      = create_regge_space(celltype, k);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat
      = create_regge_dual(celltype, k);

  std::cout << "\n\ndualmat = \n[" << dualmat << " ]\n";

  apply_dualmat_to_basis(wcoeffs, dualmat);
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Regge::tabulate(int nderiv,
                const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      expansion_basis
      = PolynomialSet::tabulate(_cell_type, _degree, nderiv, pts);

  const int psize = expansion_basis[0].cols();
  const int ndofs = _coeffs.rows();

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(expansion_basis.size());

  // Number of components in symmetric tensor of given dimension
  int nc = tdim * (tdim + 1) / 2;

  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    dresult[p].resize(pts.rows(), ndofs * tdim);
    for (int j = 0; j < nc; ++j)
      dresult[p].block(0, ndofs * j, pts.rows(), ndofs)
          = expansion_basis[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
  }

  return dresult;
}
//-----------------------------------------------------------------------------
