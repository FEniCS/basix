
#include <Eigen/Dense>
#include "polynomial.h"

// Evaluates the nth jacobi polynomial with weight parameters (a, 0)
Polynomial compute_jacobi(int a, int n);

// points
Eigen::ArrayXd compute_gauss_jacobi_points(double a, int m);

// rule (points and weights)
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a, int m);
