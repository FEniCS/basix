# Copyright (c) 2020 Chris Richardson & Matthew Scroggs
# FEniCS Project
# SPDX-License-Identifier: MIT

set(HEADERS_elements
  ${CMAKE_CURRENT_SOURCE_DIR}/lagrange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nce-rtc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/brezzi-douglas-marini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nedelec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/raviart-thomas.h
  ${CMAKE_CURRENT_SOURCE_DIR}/regge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/crouzeix-raviart.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bubble.h
  PARENT_SCOPE)

set(CPP_elements
  ${CMAKE_CURRENT_SOURCE_DIR}/lagrange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nce-rtc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/brezzi-douglas-marini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nedelec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/raviart-thomas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/regge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/crouzeix-raviart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bubble.cpp
  PARENT_SCOPE)
