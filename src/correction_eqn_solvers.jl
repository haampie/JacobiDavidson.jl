abstract CorrectionSolver

type gmres_solver <: CorrectionSolver
  iterations::Int
  tolerance::Float64
  gmres_solver(;iterations::Int = 5, tolerance::Float64 = 1e-6) = new(iterations, tolerance)
end

type exact_solver <: CorrectionSolver
end

function solve_correction{T <: AbstractLinearMap}(solver::gmres_solver, A::T, θ, u::AbstractVector)
  n = size(A, 1)

  # Define the residual mapping
  R = LinearMap(x -> A * x - θ * x, nothing, n)

  # Projection Cⁿ → Cⁿ ∖ span {u}: Px = (I - uu')x
  P = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)

  # Coefficient matrix A - θI restricted map: Cⁿ ∖ span {u} -> Cⁿ ∖ span {u}
  C = P * R * P

  # Residual
  r = R * u

  gmres(C, -r, max_iter = solver.iterations, ɛ = solver.tolerance)
end

function solve_correction{T <: AbstractLinearMap}(solver::exact_solver, A::T, θ, u::AbstractVector)
  # The exact solver is mostly useful for testing Jacobi-Davidson
  # method itself and should result in quadratic convergence.
  # However, in general the correction equation should be solved
  # only approximately in a fixed number of GMRES / BiCGStab
  # iterations to reduce computation costs.

  # This method is *not* matrix-free (TODO: assert this or specify solve_correction)

  # Here we solve the augmented system
  # [A - θI, u; u' 0][t; y] = [-r; 0] for t,
  # which is equivalent to solving (A - θI) t = -r
  # for t ⟂ u.

  n = size(A, 1)
  r = A * u - θ * u
  Ã = [(A.lmap - θ * speye(n)) u; u' 0]
  rhs = [-r; 0]
  (Ã \ rhs)[1 : n]
end

function solve_deflated_correction(solver::exact_solver, A, θ, X::AbstractMatrix, u::AbstractVector, r::AbstractVector)
  # The exact solver is mostly useful for testing Jacobi-Davidson
  # method itself and should result in quadratic convergence.
  # However, in general the correction equation should be solved
  # only approximately in a fixed number of GMRES / BiCGStab
  # iterations to reduce computation costs.

  # This method is *not* matrix-free (TODO: assert this or specify solve_correction)

  # Here we solve the augmented system
  # [A - θI, [X, u]; [X, u]' 0][t; y] = [-r; 0] for t,
  # which is equivalent to solving (A - θI) t = -r
  # for t ⟂ X and t ⟂ u.

  n = size(A, 1)
  Q = [X u]
  m = size(Q, 2)
  Ã = [(A - θ * speye(n)) Q; Q' zeros(m, m)]
  rhs = [-r; zeros(m, 1)]
  (Ã \ rhs)[1 : n]
end

function solve_deflated_correction(solver::gmres_solver, A, θ, X::AbstractMatrix, u::AbstractVector, r::AbstractVector)
  n = size(A, 1)

  # Define the residual mapping
  R = LinearMap(x -> A * x - θ * x, nothing, n)

  # Projection Cⁿ → Cⁿ ∖ span {u}: P1x = (I - uu')x
  P1 = LinearMap(x -> x - dot(u, x) * u, nothing, n; ishermitian = true)

  # Projection Cⁿ → Cⁿ ∖ span {X}: P2x = (I - XX')x
  P2 = LinearMap(x -> x - X * (X' * x), nothing, n; ishermitian = true)

  # Coefficient matrix A - θI restricted map: Cⁿ ∖ span {Q} -> Cⁿ ∖ span {Q}
  C = P1 * P2 * R

  x, _ = gmres(C, -r, outer = 1, restart = solver.iterations, tol = solver.tolerance * norm(r))

  x
end
