module Tst

import LinearMaps: LinearMap

using JacobiDavidson:
  exact_solver,
  gmres_solver,
  jacobi_davidson,
  jacobi_davidson_hermetian,
  LM, SM, Near

function alright_conditioned_matrix_with_separated_eigs(n::Int = 100)
  off_diag₁ = rand(n - 1)
  off_diag₂ = rand(n - 1)

  diags = (off_diag₁, linspace(3, 100, n), off_diag₂)
  spdiagm(diags, (-1, 0, 1))
end

function testing(;krylov = 10, expansions = 5)
  n = 300
  A = alright_conditioned_matrix_with_separated_eigs(n)
  B = LinearMap(A)

  exact = exact_solver()
  gmres = gmres_solver()
  target = Near(10.0 + 0.0im)

  @time jacobi_davidson(B, exact, krylov, expansions = expansions, target = target)
  @time jacobi_davidson(B, gmres, krylov, expansions = expansions, target = target)
end

function hermetian_example(; n = 20, max = 20)
  A = Symmetric(sprand(n, n, .2) + 5.0 * speye(n))

  exact = exact_solver()

  D, X, res = jacobi_davidson_hermetian(LinearMap(A, isposdef = false), exact)

  X, D, res
end
end
