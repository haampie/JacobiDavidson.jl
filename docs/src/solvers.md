# Solvers for the correction equation

At this point preconditioned GMRES and BiCGStabl(l) are available as iterative methods to solve the correction equation. Allocations for these methods are done only once during initialization.

## BiCGStab(l)

BiCGStab(l) is a non-optimal Krylov subspace method, but is of interest because it has a fixed amount of operations per iteration:

```julia
solver = bicgstabl_solver(n, max_mv_products = 10, l = 2)
```

## GMRES

GMRES selects the minimal residual solution from a Krylov subspace. We use GMRES without restarts, since we assume only a few iterations are performed.

```julia
solver = gmres_solver(n, iterations = 5)
```

## Preconditioning

Preconditioners can be used to improve the iterative method that solves the correction equation approximately. Although Jacobi-Davidson can be implemented with a variable or flexible preconditioner that changes each iteration, it is often more efficient to construct a fixed preconditioner for $(A - \tau B)$ or $(A - \tau I)$ for JDQZ and JDQR respectively. The motivation is that the preconditioner has to be deflated with the converged Schur vectors, which can be performed just once when the preconditioner is kept fixed.

Preconditioners `P` are expected to implement `ldiv!(P, x)` which performs `x = P \ x` in-place.
