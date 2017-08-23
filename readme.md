# Jacobi-Davidson

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://haampie.github.io/JacobiDavidson.jl/latest)

An implementation of Jacobi-Davidson in Julia.

## Example

We generate two random complex matrices A and B and use JDQZ to find the eigenvalues λ of the generalized eigenvalue problem Ax = λBx.

```julia
using JacobiDavidson, Plots

function run(n = 1000)
  A = 2 * speye(Complex128, n) + sprand(Complex128, n, n, 1 / n)
  B = 2 * speye(Complex128, n) + sprand(Complex128, n, n, 1 / n)

  # Find all eigenvalues with a direct method
  values = eigvals(full(A), full(B))

  target = Near(1.5 - 0.7im)

  schur, residuals = jdqz(
    A, B,
    gmres_solver(n, iterations = 7),
    target = target,
    pairs = 7,
    min_dimension = 10,
    max_dimension = 15,
    max_iter = 300,
    verbose = true
  )

  # The eigenvalues found by Jacobi-Davidson
  found = schur.alphas ./ schur.betas

  # 
  p1 = scatter(real(values), imag(values), label = "eig")
  scatter!(real(found), imag(found), marker = :+, label = "jdqz")
  scatter!([real(target.τ)], [imag(target.τ)], marker = :star, label = "Target")

  p2 = plot(residuals, yscale = :log10, marker = :auto, label = "Residual norm")

  p1, p2
end
```

The first plot shows the full spectrum, together with the target we have set and the seven converged eigenvalues:

![Eigenvalues found](https://haampie.github.io/JacobiDavidson.jl/latest/found.png)

The second plot shows the residual norm of Ax - λBx during the iterations:

![Residual norm](https://haampie.github.io/JacobiDavidson.jl/latest/residualnorm.png)
