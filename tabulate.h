
#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

// Tabulate line basis functions (Legendre polynomials) at given points
std::vector<Polynomial> tabulate_line(int n);

// Tabulate Dubiner triangle basis functions at given points
std::vector<Polynomial> tabulate_triangle(int n);

// Tabulate tetrahedron basis functions at given points
std::vector<Polynomial> tabulate_tetrahedron(int n);
