=======================================
The ordering of Basix's polynomial sets
=======================================

When defining finite elements, Basix uses orthonormal polyomials to represent
a basis of the space spanned by the finite element functions. This page
details the order in which these polynomials are stored.

In each example below, we give a list of the monomials that appear for the
first time in each term. The actual orthogonal polynomials will be linear
combinations of the monomial and all the previous monomials. For example,
the monomials

:math:`1`,
:math:`x`,
:math:`x^2`,
:math:`x^3`,
...

are given below for an interval. The orthonormal polynomials on an interval,
however, are

:math:`1`,
:math:`\sqrt{3}(2x-1)`,
:math:`\sqrt{5}(6x^2-6x+1)`,
:math:`\sqrt{6}(20x^3 - 30x^2 + 12x - 1)`,
...


Interval
========
On an interval, Basix orders its polynomials by lowest degree first, ie

:math:`1`,
:math:`x`,
:math:`x^2`,
:math:`x^3`,
...

Triangle
========
On a triangle, Basix orders its polynomials by degree (lowest first).
Polynomials of the same degree are ordered so that higher powers of :math:`y`
appear first, ie

:math:`1`,
:math:`y`,
:math:`x`,
:math:`y^2`,
:math:`xy`,
:math:`x^2`,
:math:`y^3`,
:math:`xy^2`,
:math:`x^2y`,
:math:`x^3`,
...

Tetrahedron
===========
On a tetrahedron, Basix orders its polynomials by degree (lowest first).
Polynomials of the same degree are ordered so that higher powers of :math:`z`
appear first then higher powers of :math:`y`, ie

:math:`1`,
:math:`z`,
:math:`y`,
:math:`x`,
:math:`z^2`,
:math:`yz`,
:math:`xz`,
:math:`y^2`,
:math:`xy`,
:math:`x^2`,
:math:`z^3`,
:math:`yz^2`,
:math:`xz^2`,
:math:`y^2z`,
:math:`xyz`,
:math:`x^2z`,
:math:`y^3`,
:math:`xy^2`,
:math:`x^2y`,
:math:`x^3`,
...

Quadrilateral
=============
On a quadrilateral, we take a tensor product of the polynomials on an
interval with :math:`y` as the variable and those with :math:`x` as the variable.
The resulting ordering has polynomials containing only :math:`y` first
(low-to-high degree), then these polynomials times :math:`x`, then the same
polynomials times :math:`x^2`, and so on; ie

:math:`1`,
:math:`y`,
:math:`y^2`,
:math:`y^3`,
:math:`x`,
:math:`xy`,
:math:`xy^2`,
:math:`xy^3`,
:math:`x^2`,
:math:`x^2y`,
:math:`x^2y^2`,
:math:`x^2y^3`,
:math:`x^3`,
:math:`x^3y`,
:math:`x^3y^2`,
:math:`x^3y^3`

Hexahedron
==========
On a hexahedron, we take a tensor product of the polynomials on an interval
with :math:`z` as the variables and the polynomials on a quadrilateral with
:math:`(x,y)` as the variables. The resulting ordering has polynomials containing
only :math:`z` first (low-to-high degree) then these polynomials times :math:`y`,
then the same polynomials times :math:`y^2`, then :math:`y^3` and so on (following
the order on a quadrilateral); ie

:math:`1`,
:math:`z`,
:math:`z^2`,
:math:`z^3`,
:math:`y`,
:math:`yz`,
:math:`yz^2`,
:math:`yz^3`,
:math:`y^2`,
:math:`y^2z`,
:math:`y^2z^2`,
:math:`y^2z^3`,
:math:`y^3`,
:math:`y^3z`,
:math:`y^3z^2`,
:math:`y^3z^3`,
:math:`x`,
:math:`xz`,
:math:`xz^2`,
:math:`xz^3`,
:math:`xy`,
:math:`xyz`,
:math:`xyz^2`,
:math:`xyz^3`,
:math:`xy^2`,
:math:`xy^2z`,
:math:`xy^2z^2`,
:math:`xy^2z^3`,
:math:`xy^3`,
:math:`xy^3z`,
:math:`xy^3z^2`,
:math:`xy^3z^3`,
:math:`x^2`,
:math:`x^2z`,
:math:`x^2z^2`,
:math:`x^2z^3`,
:math:`x^2y`,
:math:`x^2yz`,
:math:`x^2yz^2`,
:math:`x^2yz^3`,
:math:`x^2y^2`,
:math:`x^2y^2z`,
:math:`x^2y^2z^2`,
:math:`x^2y^2z^3`,
:math:`x^2y^3`,
:math:`x^2y^3z`,
:math:`x^2y^3z^2`,
:math:`x^2y^3z^3`,
:math:`x^3`,
:math:`x^3z`,
:math:`x^3z^2`,
:math:`x^3z^3`,
:math:`x^3y`,
:math:`x^3yz`,
:math:`x^3yz^2`,
:math:`x^3yz^3`,
:math:`x^3y^2`,
:math:`x^3y^2z`,
:math:`x^3y^2z^2`,
:math:`x^3y^2z^3`,
:math:`x^3y^3`
:math:`x^3y^3z`,
:math:`x^3y^3z^2`,
:math:`x^3y^3z^3`

Prism
=====
On a hexahedron, we take a tensor product of the polynomials on an interval
with :math:`z` as the variables and the polynomials on a triangle with
:math:`(x,y)` as the variables. The resulting ordering has polynomials containing
only :math:`z` first (low-to-high degree) then these polynomials times :math:`y`,
then the same polynomials times :math:`x`, then :math:`y^2` and so on (following
the order on a triangle); ie

:math:`1`,
:math:`z`,
:math:`z^2`,
:math:`y`,
:math:`yz`,
:math:`yz^2`,
:math:`x`,
:math:`xz`,
:math:`xz^2`,
:math:`y^2`,
:math:`y^2z`,
:math:`y^2z^2`,
:math:`xy`,
:math:`xyz`,
:math:`xyz^2`,
:math:`x^2`,
:math:`x^2z`,
:math:`x^2z^2`,
...

Pyramid
=======
