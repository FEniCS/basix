// ====================================
// DOF permutations and transformations
// ====================================

// When using high degree finite elements on general meshes, adjustments
// may need to be made to correct for differences in the orientation of
// mesh entities on the mesh and on the reference cell. For example, in
// a degree 4 Lagrange element on a triangle, there are 3 degrees of
// freedom (DOFs) associated with each edge. If two neighbouring cells in
// a mesh disagree on the direction of the edge, they could put an
// incorrectly combine the local basis functions to give the wrong global
// basis function.

// This issue and the use of permutations and transformations to correct
// it is discussed in detail in `Construction of arbitrary order finite
// element degree-of-freedom maps on polygonal and polyhedral cell
// meshes (Scroggs, Dokken, Richardons, Wells,
// 2021) <https://arxiv.org/abs/2102.11901>`_.

// Functions to permute and transform high degree elements are
// provided by Basix. In this demo, we show how these can be used from C++.

#include <basix/finite-element.h>
#include <basix/lattice.h>
#include <iomanip>
#include <iostream>
#include <span>

using T = double;

static const std::map<basix::cell::type, std::string> type_to_name
    = {{basix::cell::type::point, "point"},
       {basix::cell::type::interval, "interval"},
       {basix::cell::type::triangle, "triangle"},
       {basix::cell::type::tetrahedron, "tetrahedron"},
       {basix::cell::type::quadrilateral, "quadrilateral"},
       {basix::cell::type::pyramid, "pyramid"},
       {basix::cell::type::prism, "prism"},
       {basix::cell::type::hexahedron, "hexahedron"}};

int main(int argc, char* argv[])
{
  // Degree 5 Lagrange element
  // =========================
  {
    std::cout << "Degree 5 Lagrange element" << std::endl;

    // Create a degree 5 Lagrange element on a triangle
    auto family = basix::element::family::P;
    auto cell_type = basix::cell::type::triangle;
    int degree = 5;
    auto variant = basix::element::lagrange_variant::equispaced;

    // Create the lagrange element
    basix::FiniteElement lagrange
        = basix::create_element<T>(family, cell_type, degree, variant,
                                   basix::element::dpc_variant::unset, false);

    // Print bools as true/false instead of 0/1
    std::cout << std::boolalpha;

    // The value of `dof_transformations_are_identity` is `false`: this tells
    // us that permutations or transformations are needed for this element.
    std::cout << "Dof transformations are identity: "
              << lagrange.dof_transformations_are_identity() << std::endl;

    // The value of `dof_transformations_are_permutations` is `true`: this
    // tells us that for this element, all the corrections we need to apply
    // permutations. This is the simpler case, and means we make the
    // orientation corrections by applying permutations when creating the
    // DOF map.
    std::cout << "Dof transformations are permutations: "
              << lagrange.dof_transformations_are_permutations() << std::endl;

    // We can apply permutations by using the matrices returned by the
    // method `base_transformations`. This method will return one matrix
    // for each edge of the cell (for 2D and 3D cells), and two matrices
    // for each face of the cell (for 3D cells). These describe the effect
    // of reversing the edge or reflecting and rotating the face.

    // For this element, we know that the base transformations will be
    // permutation matrices.
    auto [trans, tshape] = lagrange.base_transformations();

    // The matrices returned by `base_transformations` are quite large, and
    // are equal to the identity matrix except for a small block of the
    // matrix. It is often easier and more efficient to use the matrices
    // returned by the method `entity_transformations` instead.

    // `entity_transformations` returns a dictionary that maps the type
    // of entity (`"interval"`, `"triangle"`, `"quadrilateral"`) to a
    // matrix describing the effect of permuting that entity on the DOFs
    // on that entity.
    auto entity_transformation = lagrange.entity_transformations();

    // For this element, we see that this method returns one matrix for
    // an interval: this matrix reverses the order of the four DOFs
    // associated with that edge.

    // In orders to work out which DOFs are associated with each edge,
    // we use the attribute `entity_dofs`. For example, the following can
    // be used to see which DOF numbers are associated with edge (dim 1)
    // number 2:
    int dim = 1;
    int edge_num = 2;
    std::vector<int> entity_dofs = lagrange.entity_dofs()[dim][edge_num];
    std::cout << std::endl << "Entity dofs of Edge number 2: ";
    for (const auto dof : entity_dofs)
      std::cout << dof << " ";
  }
  // Degree 2 Lagrange element
  // =========================
  {
    // For a degree 2 Lagrange element, no permutations or transformations
    // are needed.
    std::cout << std::endl
              << std::endl
              << "Degree 2 Lagrange element" << std::endl;
    auto family = basix::element::family::P;
    auto cell_type = basix::cell::type::triangle;
    int degree = 2;
    auto variant = basix::element::lagrange_variant::equispaced;

    // Create the lagrange element
    auto lagrange
        = basix::create_element<T>(family, cell_type, degree, variant,
                                   basix::element::dpc_variant::unset, false);

    // We can verify this by checking that`dof_transformations_are_identity` is
    // `True`. To confirm that the transformations are identity matrices, we
    // also print the base transformations.
    std::cout << "Dof transformations are identity: "
              << lagrange.dof_transformations_are_identity() << std::endl;

    std::cout << "Dof transformations are permutations: "
              << lagrange.dof_transformations_are_permutations() << std::endl;
  }

  // Degree 2 Nédélec element
  // ========================
  {
    std::cout << std::endl
              << std::endl
              << "Degree 2 Nedelec element" << std::endl;
    // For a degree 2 Nédélec (first kind) element on a tetrahedron, the
    // corrections are not all permutations, so both
    // `dof_transformations_are_identity` and
    // `dof_transformations_are_permutations` are `False`.

    auto family = basix::element::family::N1E;
    auto cell_type = basix::cell::type::tetrahedron;
    int degree = 2;
    auto nedelec = basix::create_element<T>(
        family, cell_type, degree, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    std::cout << "Dof transformations are identity: "
              << nedelec.dof_transformations_are_identity() << std::endl;
    std::cout << "Dof transformations are permutations: "
              << nedelec.dof_transformations_are_permutations() << std::endl;

    // For this element, `entity_transformations` returns a map
    // with two entries: a matrix for an interval that describes
    // the effect of reversing the edge; and an array of two matrices
    // for a triangle. The first matrix for the triangle describes
    // the effect of rotating the triangle. The second matrix describes
    // the effect of reflecting the triangle.

    // For this element, the matrix describing the effect of rotating
    // the triangle is

    // .. math::
    //    \left(\begin{array}{cc}-1&-1\\1&0\end{array}\right).

    // This is not a permutation, so this must be applied when
    // assembling a form and cannot be applied to the DOF numbering in
    // the DOF map.
    const auto entity__transformation = nedelec.entity_transformations();

    // To demonstrate how these transformations can be used, we create a
    // lattice of points where we will tabulate the element.

    const auto [pdata, shape] = basix::lattice::create<T>(
        cell_type, 5, basix::lattice::type::equispaced, true);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        points(pdata.data(), shape);
    int num_points = points.extent(0);

    // If (for example) the direction of edge 2 in the physical cell
    // does not match its direction on the reference, then we need to
    // adjust the tabulated data.

    const auto [original_data, orig_shape] = nedelec.tabulate(0, points);
    auto [mod_data, mod_shape] = nedelec.tabulate(0, points);
    std::span<T> data(mod_data.data(), mod_data.size());

    // If the direction of edge 2 in the physical cell is reflected, it
    // has cell permutation info `....000010` so (from right to left):
    //      - edge 0 is not permuted (0)
    //      - edge 1 is reflected (1)
    //      - edge 2 is not permuted (0)
    //      - edge 3 is not permuted (0)
    //      - edge 4 is not permuted (0)
    //      - edge 5 is not permuted (0)

    int cell_info = 0b000010;
    nedelec.T_apply(data, num_points, cell_info);
  }

  return 0;
}
