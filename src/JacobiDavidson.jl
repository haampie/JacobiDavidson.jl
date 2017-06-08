module JacobiDavidson

using SugarBLAS

import LinearMaps: AbstractLinearMap, LinearMap

include("arnoldi.jl")
include("correction_eqn_solvers.jl")
include("eigenvalue_targets.jl")

export jacobi_davidson, jacobi_davidson_hermetian, jacobi_davidson_nonhermetian, jacobi_davidson_harmonic
export gmres_solver
export exact_solver
export Near, LM, SM, LR, SR

include("jacobi_davidson.jl")
include("jacobi_davidson_hermetian.jl")
include("jacobi_davidson_nonhermetian.jl")
include("jacobi_davidson_harmonic.jl")
end
