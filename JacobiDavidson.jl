module JacobiDavidson

import LinearMaps: AbstractLinearMap, LinearMap

include("correction_eqn_solvers.jl")
include("jacobi_davidson.jl")

function some_well_conditioned_matrix(n::Int = 100)
  off_diag₁ = rand(n - 1)
  off_diag₂ = rand(n - 1)

  diags = (off_diag₁, linspace(3, 100, n), off_diag₂)
  spdiagm(diags, (-1, 0, 1))
end

function testing(;krylov = 10, expansions = 5)
  n = 300
  A = some_well_conditioned_matrix(n)

  d = eigs(A)
  @show d[1]

  B = LinearMap(A)

  jacobi_davidson(B, krylov, expansions = expansions)
end

export some_well_conditioned_matrix
export testing
export jacobi_davidson

end