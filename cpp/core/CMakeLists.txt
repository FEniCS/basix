# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

set(HEADERS_core
  ${CMAKE_CURRENT_SOURCE_DIR}/lattice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dof-permutations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mappings.h
  ${CMAKE_CURRENT_SOURCE_DIR}/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cell.h
  PARENT_SCOPE)

set(CPP_core
  ${CMAKE_CURRENT_SOURCE_DIR}/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dof-permutations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mappings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/element-families.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quadrature.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cell.cpp
  PARENT_SCOPE)
