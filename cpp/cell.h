// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>

#pragma once

class Cell
{
  /// Class holding information about reference cells
  /// including their topological and geometric data.

public:
  /// Cell type
  enum class Type
  {
    point,
    interval,
    triangle,
    tetrahedron,
    quadrilateral,
    hexahedron,
    prism,
    pyramid
  };

  /// Constructor
  /// Create a Cell of given type
  /// @param celltype Cell::Type enum
  Cell(Type celltype);

  /// Cell geometry
  /// @return Set of vertex points of the cell
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  geometry() const
  {
    return _geometry;
  }

  /// Sub-entity a cell, given by topological dimension and index
  /// @param dim Dimension of sub-entity
  /// @param index Local index of sub-entity
  /// @return Set of vertex points of the sub-entity
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  sub_entity_geometry(int dim, int index) const;

  /// Create a lattice of points on the cell, equispaced along each edge
  /// optionally including the outer surface points
  /// @param n number of points in each direction
  /// @param exterior If set, includes outer boundaries
  /// @return Set of points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_lattice(int n, bool exterior) const;

  /// Get the topological dimension for a given cell type
  /// @param cell_type Cell type
  /// @return the topological dimension
  static int topological_dimension(Cell::Type cell_type);

private:
  // Type of cell as defined by enum Cell::Type
  Type _type;

  // Cell vertex geometrical points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _geometry;

  // Cell topology, list of vertices for each dimension (0, 1, ... tdim)
  // Size of vector = (topological_dimension + 1)
  std::vector<
      Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      _topology;
};
