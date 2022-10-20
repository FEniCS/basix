// Copyright (c) 2022 Susanne Claus
// FEniCS Project
// SPDX-License-Identifier:    MIT

// This file defines the C-API functions listed in cwrapper.h 
// This is a C++ file that defines C calls 
#include <basix/finite-element.h>
#include <basix/cell.h>
#include <span>
#include "cwrapper.h"

extern "C" {

basix_element* basix_element_create(const int basix_family, const int basix_cell_type, const int degree, 
                                    const int lagrange_variant, const int dpc_variant, const bool discontinuous)
{
    //Specify which element is needed
    basix::element::family family = static_cast<basix::element::family>(basix_family); 
    basix::cell::type cell_type = static_cast<basix::cell::type>(basix_cell_type); 
    int k = degree;
    basix::element::lagrange_variant lvariant = static_cast<basix::element::lagrange_variant>(lagrange_variant);
    basix::element::dpc_variant dvariant = static_cast<basix::element::dpc_variant>(dpc_variant);

    //Create C++ basix element object
    basix::FiniteElement finite_element = basix::create_element(family, cell_type, k, lvariant,dvariant,discontinuous);

    basix_element* element = reinterpret_cast<basix_element*>(new basix::FiniteElement(finite_element));

    return element; 
}

void basix_element_destroy(basix_element *element)
{
    delete reinterpret_cast<basix::FiniteElement*>(element);
}

void basix_element_tabulate_shape(basix_element *element, const unsigned int num_points, 
                                  const int nd, int* cshape)
{
    //Determine shape of tabulated values to allocate sufficient memory 
    std::array<std::size_t, 4> shape = reinterpret_cast<basix::FiniteElement*>(element)->tabulate_shape(nd, num_points);

    //Copy shape values 
    std::copy(shape.begin(),shape.end(),cshape);
}

void basix_element_tabulate(basix_element *element, const unsigned int gdim, const double* points,
                            const unsigned int num_points, const int nd, 
                            double* table_data, long unsigned int table_data_size)
{
    basix::FiniteElement* finite_element = reinterpret_cast<basix::FiniteElement*>(element);

    //Create span views on points and table values
    std::span<const double> points_view{points,num_points*gdim};
    std::span<double> basis{table_data, table_data_size};
    std::array<std::size_t, 2> xshape{num_points,gdim};
    
    finite_element->tabulate(nd, points_view, xshape, basis);
}

}
