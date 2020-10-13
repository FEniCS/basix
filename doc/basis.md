## Expansion Polynomial Sets

TODO: Explain background here.

### Interval

The Legendre Polynomial is a special case of a [Jacobi Polynomial](https://en.wikipedia.org/wiki/Jacobi_polynomials) $P_n^{\alpha, \beta}$, with the weights $(\alpha, \beta) = (0, 0)$. Using the Jacobi recurrence relation, we have:

$$ n P^{0,0}_n(x) = (2n - 1) x P^{0,0}_{n - 1}(x) - (n-1) P^{0,0}_{n-2}(x) $$

starting with $P^{0,0}_0 = 1$. Legendre polynomials are orthogonal when integrated over the interval [-1, 1], and are used as the expansion polynomial set for a reference line interval. For a given set of points, the recurrence relation can be used to compute the polynomial set up to arbitrary order.

### Triangle - Dubiner's basis
See [Sherwin & Karniadakis IJNME 38 3775-3802 (1995)](https://doi.org/10.1016/0045-7825(94)00745-9).

An orthogonal set over a triangle can be obtained with a change of variable to $\zeta = 2{1 + x\over 1- y} - 1$, and a modified polynomial. The polynomial basis becomes:
$$ Q_{p,q} = P^{0,0}_p(\zeta) \left(1-y\over 2\right)^p\ P_q^{2p+1, 0}(y) $$
with a Legendre Polynomial of $\zeta$ and a weighted Jacobi Polynomial of $y$. In order to calculate these without actually multiplying together the two polynomials directly, we can first compute $Q_{p, 0}$ using a recurrence relation (since $P_0^{2p + 1, 0} = 1$).

$$ p Q_{p,0} = (2p-1)(2x + y+1)Q_{p-1,0} - (p-1)(1-y)^2 Q_{p-2, 0}$$

Subsequently, we can calculate $Q_{p,q}$ by building up from $Q_{p,0}$ with another recurrence relation:
$$ Q_{p,q} = (a_0 y + a_1) Q_{p, q-1} + a_2 Q_{p, q-2} $$
where $a_0, a_1, a_2$ are the coefficients in the Jacobi recurrence relation with $(\alpha,\beta) = (2p+1, 0)$, see [Wikipedia](https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations). Note that $a_2 = 0$ when $q < 2$.

### Extension to tetrahedron
See [Sherwin & Karniadakis IJNME 38 3775-3802 (1995)](https://doi.org/10.1016/0045-7825(94)00745-9).
Let $\zeta = 2{1+x\over y+z} + 1$ and $\xi = 2{1+y\over 1-z} - 1$.
$$ Q_{p, q, r} = P^{0,0}_p(\zeta)\left(y+z\over 2\right)^p\ P_q^{2p+1, 0}(\xi)\left(1-z\over 2\right)^q\ P_r^{2(p+q+1), 0}(z)$$

This can similarly be built up by first computing $Q_{p,0,0}$, then $Q_{p, q, 0}$ and finally $Q_{p, q, r}$ with recurrence relations on each axis.

### Other shapes
The same principles can be applied to quadrilateral, hexahedral, pyramid and prism elements. The polynomial sets for quadrilateral and hexahedral elements can be formed directly by the product of line interval polynomial sets, and the prism by a triangle and line interval.
The pyramid element is best calculated in the same way as the tetrahedron, using recurrence relations on each axis.

For the pyramid, $\zeta_x = 2{1+x\over 1-z} - 1$, $\zeta_y = 2{1+y\over 1-z} - 1$.
$$Q_{p, q, r} = P^{0,0}_p(\zeta_x) P^{0,0}_q(\zeta_y) \left(1-z\over 2\right)^{(p+q)} P_r^{2(p+q+1), 0}(z)$$

### Normalisation
Multiply by $\sqrt{p+1/2}$ etc.

## Spatial derivatives of the expansion sets

Recurrence relations can also be used to find the derivatives of the polynomials at given points. For example, the line interval has a first derivative given by:

$$ n P'_n(x) = (2n - 1) \left(P_{n-1}(x) + x P'_{n - 1}(x)\right) + (n-1) P'_{n-2}(x) $$
and in general for the $k$-th derivative:
$$ n P^k_n(x) = (2n - 1) \left(k P^{k-1}_{n-1}(x) + x P^k_{n - 1}(x)\right) + (n-1) P^k_{n-2}(x) $$
This is now a recurrence relation in both $n$ and $k$.
Similar recurrence relations can be obtained for the derivatives of all the polynomial sets on the other shapes. Care must be taken with quadratic terms, and cross-terms in two and three dimensions.

## Scaling
The Legendre Polynomials are orthogonal when integrated over [-1, 1], but we are interested in domains over [0, 1]. A simple scaling $x' = 2x - 1$ maps between the two domains, and often just results in some factors of 2 in various places.
