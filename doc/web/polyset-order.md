# The ordering of Basix's polynomial sets
When defining finite elements, Basix uses orthonormal polyomials to represent
a basis of the space spanned by the finite element functions. This page
details the order in which these polynomials are stored.

In each example below, we give a list of the monomials that appear for the
first time in each term. The actual orthogonal polynomials will be linear
combinations of the monomial and all the previous monomials. For example,
the monomials

\\(1\\),
\\(x\\),
\\(x^2\\),
\\(x^3\\),
...

are given below for an interval. The orthonormal polynomials on an interval,
however, are

\\(1\\),
\\(\\sqrt{3}(2x-1)\\),
\\(\\sqrt{5}(6x^2-6x+1)\\),
\\(\\sqrt{6}(20x^3 - 30x^2 + 12x - 1)\\),
...


### Interval
On an interval, Basix orders its polynomials by lowest degree first, ie

\\(1\\),
\\(x\\),
\\(x^2\\),
\\(x^3\\),
...

### Triangle
On a triangle, Basix orders its polynomials by degree (lowest first).
Polynomials of the same degree are ordered so that higher powers of \\(y\\)
appear first, ie

\\(1\\),
\\(y\\),
\\(x\\),
\\(y^2\\),
\\(xy\\),
\\(x^2\\),
\\(y^3\\),
\\(x^2y\\),
\\(xy^2\\),
\\(x^3\\),
...

### Tetrahedron
On a tetrahedron, Basix orders its polynomials by degree (lowest first).
Polynomials of the same degree are ordered so that higher powers of \\(z\\)
appear first then higher powers of \\(y\\), ie

\\(1\\),
\\(z\\),
\\(y\\),
\\(x\\),
\\(z^2\\),
\\(yz\\),
\\(xz\\),
\\(y^2\\),
\\(xy\\),
\\(x^2\\),
\\(z^3\\),
\\(yz^2\\),
\\(xz^2\\),
\\(y^2z\\),
\\(xyz\\),
\\(x^2z\\),
\\(y^3\\),
\\(xy^2\\),
\\(x^2y\\),
\\(x^3\\),
...

### Quadrilateral
On a quadrilateral, we take a tensor product of the polynomials on an
interval with \\(y\\) as the variable and those with \\(x\\) as the variable.
The resulting ordering has polynomials containing only \\(y\\) first
(low-to-high degree), then these polynomials times \\(x\\), then the same
polynomials times \\(x^2\\), and so on; ie

\\(1\\),
\\(y\\),
\\(y^2\\),
\\(y^3\\),
\\(x\\),
\\(xy\\),
\\(xy^2\\),
\\(xy^3\\),
\\(x^2\\),
\\(x^2y\\),
\\(x^2y^2\\),
\\(x^2y^3\\),
\\(x^3\\),
\\(x^3y\\),
\\(x^3y^2\\),
\\(x^3y^3\\)

### Hexahedron
On a hexahedron, we take a tensor product of the polynomials on an interval
with \\(z\\) as the variables and the polynomials on a quadrilateral with
\\((x,y)\\) as the variables. The resulting ordering has polynomials containing
only \\(z\\) first (low-to-high degree) then these polynomials times \\(y\\),
then the same polynomials times \\(y^2\\), then \\(y^3\\) and so on (following
the order on a quadrilateral); ie

\\(1\\),
\\(z\\),
\\(z^2\\),
\\(z^3\\),
\\(y\\),
\\(yz\\),
\\(yz^2\\),
\\(yz^3\\),
\\(y^2\\),
\\(y^2z\\),
\\(y^2z^2\\),
\\(y^2z^3\\),
\\(y^3\\),
\\(y^3z\\),
\\(y^3z^2\\),
\\(y^3z^3\\),
\\(x\\),
\\(xz\\),
\\(xz^2\\),
\\(xz^3\\),
\\(xy\\),
\\(xyz\\),
\\(xyz^2\\),
\\(xyz^3\\),
\\(xy^2\\),
\\(xy^2z\\),
\\(xy^2z^2\\),
\\(xy^2z^3\\),
\\(xy^3\\),
\\(xy^3z\\),
\\(xy^3z^2\\),
\\(xy^3z^3\\),
\\(x^2\\),
\\(x^2z\\),
\\(x^2z^2\\),
\\(x^2z^3\\),
\\(x^2y\\),
\\(x^2yz\\),
\\(x^2yz^2\\),
\\(x^2yz^3\\),
\\(x^2y^2\\),
\\(x^2y^2z\\),
\\(x^2y^2z^2\\),
\\(x^2y^2z^3\\),
\\(x^2y^3\\),
\\(x^2y^3z\\),
\\(x^2y^3z^2\\),
\\(x^2y^3z^3\\),
\\(x^3\\),
\\(x^3z\\),
\\(x^3z^2\\),
\\(x^3z^3\\),
\\(x^3y\\),
\\(x^3yz\\),
\\(x^3yz^2\\),
\\(x^3yz^3\\),
\\(x^3y^2\\),
\\(x^3y^2z\\),
\\(x^3y^2z^2\\),
\\(x^3y^2z^3\\),
\\(x^3y^3\\)
\\(x^3y^3z\\),
\\(x^3y^3z^2\\),
\\(x^3y^3z^3\\)

### Prism
On a hexahedron, we take a tensor product of the polynomials on an interval
with \\(z\\) as the variables and the polynomials on a triangle with
\\((x,y)\\) as the variables. The resulting ordering has polynomials containing
only \\(z\\) first (low-to-high degree) then these polynomials times \\(y\\),
then the same polynomials times \\(x\\), then \\(y^2\\) and so on (following
the order on a triangle); ie

\\(1\\),
\\(z\\),
\\(z^2\\),
\\(y\\),
\\(yz\\),
\\(yz^2\\),
\\(x\\),
\\(xz\\),
\\(xz^2\\),
\\(y^2\\),
\\(y^2z\\),
\\(y^2z^2\\),
\\(xy\\),
\\(xyz\\),
\\(xyz^2\\),
\\(x^2\\),
\\(x^2z\\),
\\(x^2z^2\\),
...

### Pyramid
