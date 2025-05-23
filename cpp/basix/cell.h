// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <array>
#include <concepts>
#include <utility>
#include <vector>

/// Information about reference cells

/// This namespace include functions that can be used to obtain
/// geometric and topological information about reference cells
namespace basix::cell
{

/// Cell type
enum class type : int
{
  point = 0,
  interval = 1,
  triangle = 2,
  tetrahedron = 3,
  quadrilateral = 4,
  hexahedron = 5,
  prism = 6,
  pyramid = 7
};

/// Cell geometry
/// @param celltype Cell Type
/// @return (0) Vertex point data of the cell and (1) the shape of the
/// data array. The points are stored in row-major format and the shape
/// is is (npoints, gdim)
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
geometry(cell::type celltype);

/// Cell topology
/// @param celltype Cell Type
/// @return List of topology (vertex indices) for each dimension (0..tdim)
std::vector<std::vector<std::vector<int>>> topology(cell::type celltype);

/// Get the numbers of entities connected to each subentity of the cell.
///
/// Returns a vector of the form: output[dim][entity_n][connected_dim] =
/// [connected_entity_n0, connected_entity_n1, ...] This indicates that
/// the entity of dimension `dim` and number `entity_n` is connected to
/// the entities of dimension `connected_dim` and numbers
/// `connected_entity_n0`, `connected_entity_n1`, ...
///
/// @param celltype Cell Type
/// @return List of topology (vertex indices) for each dimension (0..tdim)
std::vector<std::vector<std::vector<std::vector<int>>>>
sub_entity_connectivity(cell::type celltype);

/// Sub-entity of a cell, given by topological dimension and index
/// @param celltype The cell::type
/// @param dim Dimension of sub-entity
/// @param index Local index of sub-entity
/// @return Set of vertex points of the sub-entity. Shape is (npoints, gdim)
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
sub_entity_geometry(cell::type celltype, int dim, int index);

/// Number of sub-entities of a cell by topological dimension
/// @param celltype The cell::type
/// @param dim Dimension of sub-entity
/// @return The number of sub-entities of the given dimension
int num_sub_entities(cell::type celltype, int dim);

/// Get the topological dimension for a given cell type
/// @param celltype Cell type
/// @return the topological dimension
int topological_dimension(cell::type celltype);

/// Get the cell type of a sub-entity of given dimension and index
/// @param celltype Type of cell
/// @param dim Topological dimension of sub-entity
/// @param index Index of sub-entity
/// @return cell type of sub-entity
cell::type sub_entity_type(cell::type celltype, int dim, int index);

/// Get the volume of a reference cell
/// @param cell_type Type of cell
/// @return The volume of the cell
template <std::floating_point T>
T volume(cell::type cell_type);

/// Get the (outward) normals to the facets of a reference cell
/// @param cell_type Type of cell
/// @return The outward normals. Shape is (nfacets, gdim)
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
facet_outward_normals(cell::type cell_type);

/// Get the normals to the facets of a reference cell oriented using the
/// low-to-high ordering of the facet, scaled so that the length of the
/// normal is the volume of the facet
/// @param cell_type Type of cell
/// @return The normals. Shape is (nfacets, gdim)
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
scaled_facet_normals(cell::type cell_type);

/// Get the normals to the facets of a reference cell oriented using the
/// low-to-high ordering of the facet
/// @param cell_type Type of cell
/// @return The normals. Shape is (nfacets, gdim)
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
facet_normals(cell::type cell_type);

/// Get an array of bools indicating whether or not the facet normals are
/// outward pointing
/// @param cell_type Type of cell
/// @return The orientations
std::vector<bool> facet_orientations(cell::type cell_type);

/// Get the reference volumes of the facets of a reference cell
/// @param cell_type Type of cell
/// @return The volumes of the references associated with each facet
template <std::floating_point T>
std::vector<T> facet_reference_volumes(cell::type cell_type);

/// Get the types of the subentities of a reference cell
/// @param cell_type Type of cell
/// @return The subentity types. Indices are (tdim, entity)
std::vector<std::vector<cell::type>> subentity_types(cell::type cell_type);

/// Get the jacobians of a set of entities of a reference cell
/// @param cell_type Type of cell
/// @param e_dim Dimension of entities
/// @return The jacobians of the facets (flattened) and the shape (nfacets,
/// tdim, tdim - 1).
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 3>>
entity_jacobians(cell::type cell_type, std::size_t e_dim);

/// Get the jacobians of the facets of a reference cell
/// @param cell_type Type of cell
/// @return The jacobians of the facets (flattened) and the shape (nfacets,
/// tdim, tdim - 1).
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 3>>
facet_jacobians(cell::type cell_type);

/// Get the jacobians of the edegs of a reference cell
/// @param cell_type Type of cell
/// @return The jacobians of the edges (flattened) and the shape (nedges, tdim,
/// tdim-2).
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 3>>
edge_jacobians(cell::type cell_type);

} // namespace basix::cell
