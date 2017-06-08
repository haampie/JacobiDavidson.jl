# Jacobi-Davidson [WIP!]

An implementation of Jacobi-Davidson in Julia. Still very much WIP.

Todo:

 - [x] Matrix-free with LinearMaps.jl (but maybe overriding A_mul_B could work just as well)
 - [x] Targeting largest or smallest magnitude OR smallest or largest real part OR near a specified target in the complex plane
 - [x] Single eigenpair extraction
 - [ ] Adding tests
 - [ ] Preconditioning
 - [ ] Efficient implementation for Hermitian matrices (+ BiCGStab)
 - [x] Multiple eigenpairs & locking
 - [x] Implicit restart
 - [x] Convergence history / status
 - [ ] Use harmonic Ritz values rather than Ritz values for better approximations of interior eigenvalues.


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