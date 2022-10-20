// Copyright (c) 2022 Susanne Claus
// FEniCS Project
// SPDX-License-Identifier:    MIT

// Declares a simple C API wrapper around C++.

#ifndef CWRAPPER_H_
#define CWRAPPER_H_

#include <stdbool.h>

#ifdef __cplusplus
// extern C indicates to C++ compiler that the following code 
// is meant to be called by a C compiler 
// i.e. tells the compiler that C naming conventions etc should be used
extern "C" {
#endif

//@todo Could this be replaced by ufcx_finite_element? 
/// Basix element structure to map c basix element to C++ basix finite_element
/// The following int values indicate the position of the values in the corresponding 
/// enum classes in basix
typedef struct basix_element
{    
  /// Basix identifier of the family 
  int basix_family;
  /// Basix identifier of the cell type
  int basix_cell_type;
  /// Polynomial degree 
  int degree;
  /// The Lagrange variant to be passed to Basix's create_element function
  int lagrange_variant;
  /// The DPC variant to be passed to Basix's create_element function
  int dpc_variant;
  /// Indicates whether or not this is the discontinuous version of the element
  bool discontinuous;
  /// The geometric dimension
  int gdim;
} basix_element;

basix_element* basix_element_create(const int basix_family, const int basix_cell_type, const int degree, 
                                    const int lagrange_variant, const int dpc_variant, const bool discontinuous, 
                                    const unsigned int gdim);

void basix_element_destroy(basix_element *element);

void basix_element_tabulate(const basix_element *element, const double* points,
                            const unsigned int num_points, const int nd, 
                            double* table_data, long unsigned int table_data_size);

void basix_element_tabulate_shape(const basix_element *element, const unsigned int num_points, 
                                  const int nd, int* cshape);

#ifdef __cplusplus
}
#endif


#endif /*CWRAPPER_H_*/
