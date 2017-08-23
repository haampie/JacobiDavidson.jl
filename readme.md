# Jacobi-Davidson

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://haampie.github.io/JacobiDavidson.jl/latest)

An implementation of Jacobi-Davidson in Julia.


```julia
# Find an eigenpair close to 10 in the complex plane
# starting with Ritz values of a 10-dimensional
# Krylov subspace, and expanding it with 5 basis vectors
# obtained by solving the correction eqn approximately
function testing(;krylov = 10, expansions = 5)
  A = LinearMap(...)

  exact = exact_solver()
  gmres = gmres_solver(iterations = 5)
  target = Near(10.0 + 0.0im)

  λ₁, x₁ = jacobi_davidson(A, exact, krylov, expansions = expansions, target = target)
  λ₂, x₂ = jacobi_davidson(A, gmres, krylov, expansions = expansions, target = target)
end
```