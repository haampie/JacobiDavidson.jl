module JacobiDavidson

using SugarBLAS

import LinearMaps: AbstractLinearMap, LinearMap

include("correction_eqn_solvers.jl")
include("eigenvalue_targets.jl")

export jacobi_davidson, jacobi_davidson_hermetian, jacobi_davidson_nonhermetian, jacobi_davidson_harmonic, harmonic_ritz_test
export gmres_solver
export gmres
export exact_solver
export Near, LM, SM, LR, SR

include("jacobi_davidson.jl")
include("jacobi_davidson_hermetian.jl")
include("jacobi_davidson_nonhermetian.jl")
include("jacobi_davidson_harmonic.jl")
include("harmonic_ritz_test.jl")
include("gmres.jl")
end
